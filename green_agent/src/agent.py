"""
HealthcareFraudGreenAgent — domain logic.

Assessment flow:
1. Parse incoming message: extract purple_agent_url + config (num_episodes, claims_per_episode)
2. For each episode:
   a. Reset the local fraud environment
   b. Call purple agent via A2A for each claim in the episode
   c. Execute the purple agent's action in the environment
   d. Collect step rewards and outcomes
3. Aggregate scores across episodes
4. Emit results as artifact via updater
"""

import json
import logging
import sys
import os
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message

from messenger import PurpleAgentClient

# Environment is co-located in this image under /app/environment/.
# We insert /app (repo root) — NOT /app/environment — so that Python resolves the
# package as `environment.*`.  Inserting /app/environment would make `server` the
# top-level name, which collides with this file's own sibling `server.py` and also
# breaks the relative imports inside environment/server/environment.py
# (e.g. `from ..models import …` requires `environment` to be the root package).
sys.path.insert(0, "/app")
from environment.server.environment import ClaimsFraudEnvironment, EnvironmentConfig  # noqa: E402
from environment.models import ClaimAction  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_NUM_EPISODES = 3
DEFAULT_CLAIMS_PER_EPISODE = 20  # Keep short for benchmark runs


def _extract_text(message: Message) -> str:
    """Pull plain text from an A2A message."""
    for part in message.parts or []:
        if isinstance(part.root, TextPart):
            return part.root.text
    return ""


def _parse_request(text: str) -> dict[str, Any]:
    """
    Parse assessment request JSON sent by AgentBeats.

    Expected format (from scenario.toml config):
    {
        "participants": {"agent": "http://purple-agent:9009/"},
        "config": {
            "num_episodes": 3,
            "claims_per_episode": 20,
            "fraud_rate": 0.15,
            "seed": 42
        }
    }
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: treat whole text as purple agent URL (quick-test mode)
        return {
            "participants": {"agent": text.strip()},
            "config": {},
        }


class HealthcareFraudGreenAgent:
    """
    Runs healthcare fraud episodes and scores a purple agent.
    One instance per assessment context (isolated state).
    """

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        text = _extract_text(message)
        request = _parse_request(text)

        participants = request.get("participants", {})
        config = request.get("config", {})

        purple_url = participants.get("agent") or participants.get("purple_agent")
        if not purple_url:
            await updater.add_artifact(
                [TextPart(text=json.dumps({"error": "No purple agent URL provided"}))],
                name="error",
            )
            return

        num_episodes = int(config.get("num_episodes", DEFAULT_NUM_EPISODES))
        claims_per_episode = int(config.get("claims_per_episode", DEFAULT_CLAIMS_PER_EPISODE))
        fraud_rate = float(config.get("fraud_rate", 0.15))
        seed = config.get("seed", None)

        await updater.update_status(
            state=TaskState.working,
            message=new_agent_text_message(
                f"Starting {num_episodes} episodes × {claims_per_episode} claims "
                f"(fraud_rate={fraud_rate:.0%}) against {purple_url}"
            ),
        )

        purple_client = PurpleAgentClient(purple_url)
        episode_results = []

        for ep_idx in range(num_episodes):
            ep_seed = (seed + ep_idx) if seed is not None else None
            env_config = EnvironmentConfig(
                claims_per_episode=claims_per_episode,
                fraud_rate=fraud_rate,
                seed=ep_seed,
            )
            env = ClaimsFraudEnvironment(env_config)
            obs = env.reset()

            ep_rewards = []
            ep_decisions = []
            ep_correct = 0
            step = 0

            while not obs.done:
                # Call purple agent with the claim prompt
                prompt = obs.prompt or f"Claim ID: {obs.claim_id}\nAmount: ${obs.claim_amount:.2f}"
                try:
                    agent_response = await purple_client.get_decision(prompt)
                except Exception as e:
                    logger.warning(f"Purple agent error ep={ep_idx} step={step}: {e}")
                    agent_response = "Decision: APPROVE\nRationale: Error fallback.\nEvidence: N/A"

                # Execute in environment
                action = ClaimAction(response_text=agent_response)
                obs = env.step(action)

                ep_rewards.append(obs.reward or 0.0)
                ep_decisions.append(action.parsed_decision or "UNKNOWN")
                step += 1

            # Episode summary from environment state
            state = env.state
            ep_result = {
                "episode": ep_idx + 1,
                "total_reward": sum(ep_rewards),
                "mean_reward": sum(ep_rewards) / max(len(ep_rewards), 1),
                "steps": step,
                "true_positives": state.true_positives,
                "false_positives": state.false_positives,
                "true_negatives": state.true_negatives,
                "false_negatives": state.false_negatives,
                "fraud_caught_amount": state.fraud_amount_caught,
                "fraud_missed_amount": state.fraud_amount_missed,
                "investigation_cost": state.investigation_cost,
                "precision": (
                    state.true_positives / max(state.true_positives + state.false_positives, 1)
                ),
                "recall": (
                    state.true_positives / max(state.true_positives + state.false_negatives, 1)
                ),
            }
            episode_results.append(ep_result)

            await updater.update_status(
                state=TaskState.working,
                message=new_agent_text_message(
                    f"Episode {ep_idx + 1}/{num_episodes}: "
                    f"reward={ep_result['total_reward']:.2f}, "
                    f"precision={ep_result['precision']:.2f}, "
                    f"recall={ep_result['recall']:.2f}"
                ),
            )

        # Aggregate across episodes
        agg = {
            "num_episodes": num_episodes,
            "mean_total_reward": sum(r["total_reward"] for r in episode_results) / num_episodes,
            "mean_precision": sum(r["precision"] for r in episode_results) / num_episodes,
            "mean_recall": sum(r["recall"] for r in episode_results) / num_episodes,
            "total_fraud_caught": sum(r["fraud_caught_amount"] for r in episode_results),
            "total_fraud_missed": sum(r["fraud_missed_amount"] for r in episode_results),
            "episodes": episode_results,
        }

        # Primary leaderboard metric: mean_total_reward (the weighted RL objective).
        # F1 is logged as secondary — the blog post explicitly explains why F1 is the
        # wrong goal for this environment (Finding 4): it rewards coverage regardless of
        # investigation cost, whereas mean_total_reward penalises wasteful investigations.
        p = agg["mean_precision"]
        r = agg["mean_recall"]
        agg["f1"] = 2 * p * r / max(p + r, 1e-9)
        agg["primary_metric"] = agg["mean_total_reward"]  # higher = better

        result_json = json.dumps(agg, indent=2)

        # Emit as artifact (AgentBeats stores this as JSON result)
        await updater.add_artifact(
            [TextPart(text=result_json)],
            name="assessment_results",
        )

        await updater.update_status(
            state=TaskState.working,
            message=new_agent_text_message(
                f"Assessment complete. "
                f"MeanReward={agg['mean_total_reward']:.2f} (primary), "
                f"F1={agg['f1']:.3f} (secondary)"
            ),
        )
