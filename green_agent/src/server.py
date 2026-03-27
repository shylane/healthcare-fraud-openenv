"""
Green Agent Server — Healthcare Claims Fraud Detection
AgentBeats-compatible A2A server using the official a2a-sdk.

Run locally:
    uv run src/server.py --host 0.0.0.0 --port 9009

In Docker:
    docker run -p 9009:9009 healthcare-fraud-green:latest
"""

import argparse
import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_agent_card(host: str, port: int, card_url: str | None = None) -> AgentCard:
    base_url = card_url or f"http://{host}:{port}/"

    skill = AgentSkill(
        id="healthcare_fraud_eval",
        name="Healthcare Claims Fraud Evaluation",
        description=(
            "Evaluates an LLM agent's ability to detect healthcare insurance fraud "
            "across sequential claim decisions. Runs N episodes, scores each decision "
            "on correctness, rationale quality, evidence citation, and investigation "
            "budget management. Returns per-episode and aggregate scores."
        ),
        tags=["healthcare", "fraud-detection", "rl", "evaluation"],
        examples=[
            "Evaluate agent on 5 healthcare fraud detection episodes",
            "Run a 10-episode assessment with 100 claims each",
        ],
        input_modes=["text"],
        output_modes=["text"],
    )

    return AgentCard(
        name="Healthcare Claims Fraud Detection — Green Agent",
        description=(
            "AgentBeats evaluator for the OpenEnv Healthcare Claims Fraud environment. "
            "Generates synthetic healthcare claim episodes and scores LLM agents on "
            "fraud detection accuracy, reasoning quality, and budget efficiency."
        ),
        url=base_url,
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main():
    parser = argparse.ArgumentParser(description="Healthcare Fraud Green Agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", default=None,
                        help="Public URL for AgentCard (e.g. https://my-agent.hf.space/)")
    args = parser.parse_args()

    agent_card = build_agent_card(args.host, args.port, args.card_url)

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info(f"Starting Healthcare Fraud Green Agent on {args.host}:{args.port}")
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
