"""
Healthcare Claims Fraud Detection Environment Implementation.

This is the core environment that implements the OpenEnv specification
with reset(), step(), and state() methods.

Refactored for:
1. LLM-native text generation action space
2. Multi-component reward scoring
3. Proper reproducibility (seeded RNG)
4. OpenEnv type compatibility
"""

import uuid
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..models import (
    ClaimAction,
    ClaimObservation,
    ClaimState,
    DecisionType,
    InvestigationResult,
    RewardConfig,
    RewardComponents,
)
from ..claims_simulator import ClaimsSimulator, SimulatorConfig


@dataclass
class EnvironmentConfig:
    """Configuration for the Claims Fraud Environment."""

    # Episode settings
    claims_per_episode: int = 100

    # Fraud settings
    fraud_rate: float = 0.05

    # Reward configuration
    reward_config: Optional[RewardConfig] = None

    # Simulator configuration
    simulator_config: Optional[SimulatorConfig] = None

    # Optional data loader for real data (Hybrid Mode)
    data_loader: Optional[Any] = None

    # Action effectiveness (probability of correct outcome given action)
    investigate_accuracy: float = 0.95  # Investigation catches 95% of fraud
    flag_accuracy: float = 0.70  # Review catches 70% of fraud
    approve_fraud_miss_rate: float = 0.0  # Approving passes all fraud through

    # Random seed for reproducibility
    seed: Optional[int] = None

    # --- Enhanced environment settings ---

    # Investigation budget: how many INVESTIGATE actions the agent can take per episode
    investigation_budget: int = 15

    # Memory decay halflife: number of turns for investigation confidence to halve
    memory_decay_halflife: int = 20

    # Decision history window: how many recent decisions to show in prompt
    decision_history_window: int = 20

    def __post_init__(self):
        if self.reward_config is None:
            self.reward_config = RewardConfig()
        if self.simulator_config is None:
            self.simulator_config = SimulatorConfig(
                fraud_rate=self.fraud_rate,
                claims_per_episode=self.claims_per_episode,
                seed=self.seed,
            )


class ClaimsFraudEnvironment(object):
    """
    OpenEnv-compatible environment for healthcare claims fraud detection.

    This environment is designed for LLM agents. The agent receives claims
    as text prompts and must generate structured text responses containing:
    - A decision (APPROVE, FLAG_REVIEW, INVESTIGATE, DENY, REQUEST_INFO)
    - A rationale explaining the decision
    - Evidence citations from the claim data

    This requires actual LLM text generation capabilities, not just classification.

    Rewards are multi-component:
    - Decision correctness (did you make the right call?)
    - Rationale quality (is your explanation coherent?)
    - Evidence citation (did you cite relevant data?)
    - Efficiency (was this cost-effective?)

    Example:
        >>> env = ClaimsFraudEnvironment()
        >>> obs = env.reset()
        >>> prompt = obs.to_prompt()  # Get LLM-ready prompt
        >>> # LLM generates response text...
        >>> action = ClaimAction(response_text=llm_response)
        >>> obs = env.step(action)
        >>> print(f"Reward: {obs.reward}")
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """
        Initialize the environment.

        Args:
            config: Environment configuration. Uses defaults if None.
        """
        self.config = config or EnvironmentConfig()

        # Initialize seeded random number generator for reproducibility
        self._rng = random.Random(self.config.seed)

        # Initialize simulator (with optional data loader)
        self.simulator = ClaimsSimulator(
            self.config.simulator_config, data_loader=self.config.data_loader
        )

        # Episode state
        self._episode_id: str = ""
        self._step_count: int = 0
        self._claims_queue: List[Tuple[ClaimObservation, bool]] = []
        self._current_claim_idx: int = 0
        self._current_observation: Optional[ClaimObservation] = None
        self._current_is_fraud: bool = False

        # Tracking state
        self._state: Optional[ClaimState] = None

        # Last reward components (for logging/debugging)
        self._last_reward_components: Optional[RewardComponents] = None

        # Stochastic detection outcomes for the current step.
        # Set inside _calculate_decision_reward() and consumed in
        # _calculate_action_reward() to avoid passing is_fraud ground truth
        # through to outcome strings and investigation memory.
        self._last_investigation_caught: bool = True
        self._last_flag_caught: bool = True

        # The decision that was actually executed (may differ from parsed_decision
        # when an over-budget INVESTIGATE is downgraded to FLAG_REVIEW).
        # Set inside _calculate_action_reward() and read by step() for metadata.
        self._last_executed_decision: str = ""

    def reset(self) -> ClaimObservation:
        """Reset the environment and start a new episode.

        Uses **lazy claim generation**: instead of generating all claims
        upfront via ``generate_batch()``, claims are generated one-at-a-time
        in ``step()`` using ``simulator.generate_claim()``.  This allows
        provider profiles to evolve between steps.

        Returns:
            ClaimObservation with initial claim data and LLM prompt.
        """
        # Generate new episode ID
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._current_claim_idx = 0

        # Reset RNG for reproducibility within episode
        if self.config.seed is not None:
            self._rng = random.Random(self.config.seed)

        # Reset simulator state (clears histories, reseeds)
        self.simulator.reset()

        # Initialize state with enhanced fields
        self._state = ClaimState(
            episode_id=self._episode_id,
            step_count=0,
            total_claims_processed=0,
            claims_approved=0,
            claims_flagged=0,
            claims_investigated=0,
            claims_denied=0,
            claims_pending_info=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            total_claim_value_processed=0.0,
            fraud_amount_caught=0.0,
            fraud_amount_missed=0.0,
            investigation_cost=0.0,
            false_positive_cost=0.0,
            cumulative_reward=0.0,
            remaining_claims_in_batch=self.config.claims_per_episode,
            rationale_quality_sum=0.0,
            evidence_citation_sum=0.0,
            valid_response_count=0,
            invalid_response_count=0,
            # Enhanced fields
            investigation_budget=self.config.investigation_budget,
            budget_remaining=self.config.investigation_budget,
            investigation_memory={},
            decision_history=[],
            risk_score=0.0,
        )

        # Lazy generation: generate first claim only
        self._current_observation, self._current_is_fraud = self.simulator.generate_claim()

        # Compute risk score for the first claim
        self._state.risk_score = self._calculate_risk_score(self._current_observation)

        # Generate prompt for LLM (with state context)
        self._current_observation.prompt = self._current_observation.to_prompt(
            state=self._state,
            history_window=self.config.decision_history_window,
            memory_halflife=self.config.memory_decay_halflife,
        )
        self._current_observation.done = False
        self._current_observation.reward = 0.0
        self._current_observation.metadata = {
            "episode_id": self._episode_id,
            "claims_remaining": self.config.claims_per_episode,
            "step": 0,
        }

        return self._current_observation

    def step(self, action: ClaimAction) -> ClaimObservation:
        """Process the current claim with the given LLM-generated action.

        After processing, generates the **next** claim lazily via
        ``simulator.generate_claim()`` (not from a pre-generated queue).
        Also records the decision in history and updates budget/memory.

        Args:
            action: ClaimAction containing the LLM's response text.

        Returns:
            ClaimObservation with next claim, reward, and done flag.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._current_observation is None:
            raise RuntimeError("No current claim. Episode may have ended.")

        # Parse the LLM response and calculate multi-component reward
        reward_components = self._process_action(action)
        reward = reward_components.total_reward
        self._last_reward_components = reward_components

        # Update step count
        self._step_count += 1
        self._state.step_count = self._step_count

        # Update cumulative reward
        self._state.cumulative_reward += reward

        # Move to next claim
        self._current_claim_idx += 1

        # Check if episode is done
        done = self._current_claim_idx >= self.config.claims_per_episode

        if done:
            # Episode complete — return final observation
            final_obs = self._current_observation  # reuse last claim
            final_obs.done = True
            final_obs.reward = reward
            final_obs.metadata = self._get_episode_summary()
            final_obs.prompt = None  # No more prompts needed
            self._current_observation = None
            return final_obs
        else:
            # Lazy generation: generate next claim now
            self._current_observation, self._current_is_fraud = self.simulator.generate_claim()
            self._state.remaining_claims_in_batch = (
                self.config.claims_per_episode - self._current_claim_idx
            )

            # Compute risk score for new claim
            self._state.risk_score = self._calculate_risk_score(self._current_observation)

            # Generate prompt for next claim (with full state context)
            self._current_observation.prompt = self._current_observation.to_prompt(
                state=self._state,
                history_window=self.config.decision_history_window,
                memory_halflife=self.config.memory_decay_halflife,
            )
            self._current_observation.done = False
            self._current_observation.reward = reward
            self._current_observation.metadata = {
                # Use the executed decision, not the requested one — they differ when
                # an over-budget INVESTIGATE is silently downgraded to FLAG_REVIEW.
                "action_taken": self._last_executed_decision,
                "action_requested": action.parsed_decision,  # for debugging
                "reward_breakdown": reward_components.to_dict(),
                "step": self._step_count,
                "budget_remaining": self._state.budget_remaining,
            }

            return self._current_observation

    def _process_action(self, action: ClaimAction) -> RewardComponents:
        """Process the LLM action and calculate multi-component reward.

        Enhanced with budget enforcement, memory recording, and decision
        history tracking.

        Args:
            action: The LLM-generated action.

        Returns:
            RewardComponents with detailed reward breakdown.
        """
        claim = self._current_observation
        is_fraud = self._current_is_fraud
        claim_amount = claim.claim_amount
        reward_cfg = self.config.reward_config

        # Initialize reward components
        components = RewardComponents()

        # Parse the LLM response
        action.parse_response()

        # Update claim value processed
        self._state.total_claim_value_processed += claim_amount
        self._state.total_claims_processed += 1

        # Check if response is valid (parseable)
        decision = action.get_decision()
        if decision is None:
            # Invalid response - penalty
            components.parse_penalty = -reward_cfg.invalid_response_penalty
            self._state.invalid_response_count += 1

            # Default to FLAG_REVIEW for safety
            decision = DecisionType.FLAG_REVIEW
        else:
            self._state.valid_response_count += 1

        # --- Budget enforcement ---
        # If INVESTIGATE requested but budget exhausted, downgrade to FLAG_REVIEW
        budget_penalty = 0.0
        if decision == DecisionType.INVESTIGATE and self._state.budget_remaining <= 0:
            decision = DecisionType.FLAG_REVIEW
            budget_penalty = -0.5  # penalty for attempted over-budget investigation

        # Record executed decision so step() can report it accurately in metadata.
        # Without this, metadata["action_taken"] would always show the *requested*
        # action even when an over-budget downgrade silently changed it.
        self._last_executed_decision = decision.value

        # Deduct budget if INVESTIGATE is (still) the decision
        if decision == DecisionType.INVESTIGATE:
            self._state.budget_remaining -= 1

        # Calculate decision reward (correctness)
        decision_reward = self._calculate_decision_reward(
            decision, is_fraud, claim_amount, reward_cfg
        )
        components.decision_reward = decision_reward + budget_penalty

        # Calculate rationale quality reward
        rationale_reward = self._score_rationale(action, claim, is_fraud)
        components.rationale_reward = rationale_reward
        self._state.rationale_quality_sum += rationale_reward

        # Calculate evidence citation reward
        evidence_reward = self._score_evidence(action, claim)
        components.evidence_reward = evidence_reward
        self._state.evidence_citation_sum += evidence_reward

        # Calculate efficiency reward (penalize over-investigation)
        efficiency_reward = self._calculate_efficiency(decision, is_fraud, claim_amount, reward_cfg)
        components.efficiency_reward = efficiency_reward

        # --- Determine outcome string for history (uses stochastic detection result) ---
        # Bug fix: previously hard-coded "Fraud caught ✓" for any fraud+INVESTIGATE/FLAG,
        # ignoring the stochastic accuracy miss. Now uses the actual detection outcome
        # stored in self._last_investigation_caught / self._last_flag_caught.
        if is_fraud:
            if decision == DecisionType.INVESTIGATE:
                outcome = "Fraud caught ✓" if self._last_investigation_caught else "Fraud MISSED ✗"
            elif decision == DecisionType.FLAG_REVIEW:
                outcome = "Fraud caught ✓" if self._last_flag_caught else "Fraud MISSED ✗"
            elif decision == DecisionType.DENY:
                outcome = "Fraud caught ✓"  # DENY is deterministic
            else:
                outcome = "Fraud MISSED ✗"
        else:
            if decision == DecisionType.APPROVE:
                outcome = "Legit ✓"
            else:
                outcome = "False alarm"

        # For investigation memory: only record as fraud when it was actually DETECTED.
        # Storing raw is_fraud ground truth would leak the correct label to agents via
        # the memory section of the prompt on re-encounters.
        memory_is_fraud = is_fraud
        if decision == DecisionType.INVESTIGATE:
            memory_is_fraud = is_fraud and self._last_investigation_caught

        # Record this decision in history + update memory
        self._record_decision(decision, claim, memory_is_fraud, outcome)

        return components

    def _calculate_decision_reward(
        self,
        decision: DecisionType,
        is_fraud: bool,
        claim_amount: float,
        reward_cfg: RewardConfig,
    ) -> float:
        """Calculate reward based on decision correctness."""
        reward = 0.0

        if decision == DecisionType.APPROVE:
            self._state.claims_approved += 1

            if is_fraud:
                # Missed fraud - this is bad
                self._state.false_negatives += 1
                self._state.fraud_amount_missed += claim_amount
                reward = -claim_amount * reward_cfg.fraud_missed_penalty_rate
            else:
                # Correct approval
                self._state.true_negatives += 1
                reward = reward_cfg.efficient_approval_reward

        elif decision == DecisionType.FLAG_REVIEW:
            self._state.claims_flagged += 1
            self._state.investigation_cost += reward_cfg.review_cost

            # Simulate review outcome using seeded RNG
            review_catches_fraud = self._rng.random() < self.config.flag_accuracy
            self._last_flag_caught = review_catches_fraud

            if is_fraud:
                if review_catches_fraud:
                    self._state.true_positives += 1
                    self._state.fraud_amount_caught += claim_amount
                    reward = (
                        claim_amount * reward_cfg.fraud_caught_reward_rate - reward_cfg.review_cost
                    )
                else:
                    self._state.false_negatives += 1
                    self._state.fraud_amount_missed += claim_amount
                    reward = (
                        -claim_amount * reward_cfg.fraud_missed_penalty_rate
                        - reward_cfg.review_cost
                    )
            else:
                self._state.false_positives += 1
                self._state.false_positive_cost += reward_cfg.false_positive_penalty
                reward = -reward_cfg.review_cost - reward_cfg.false_positive_penalty

        elif decision == DecisionType.INVESTIGATE:
            self._state.claims_investigated += 1
            self._state.investigation_cost += reward_cfg.investigation_cost

            # Simulate investigation outcome using seeded RNG
            investigation_catches_fraud = self._rng.random() < self.config.investigate_accuracy
            self._last_investigation_caught = investigation_catches_fraud

            if is_fraud:
                if investigation_catches_fraud:
                    self._state.true_positives += 1
                    self._state.fraud_amount_caught += claim_amount
                    reward = (
                        claim_amount * reward_cfg.fraud_caught_reward_rate
                        - reward_cfg.investigation_cost
                    )
                else:
                    self._state.false_negatives += 1
                    self._state.fraud_amount_missed += claim_amount
                    reward = (
                        -claim_amount * reward_cfg.fraud_missed_penalty_rate
                        - reward_cfg.investigation_cost
                    )
            else:
                self._state.false_positives += 1
                self._state.false_positive_cost += reward_cfg.false_positive_penalty
                reward = -reward_cfg.investigation_cost - reward_cfg.false_positive_penalty

        elif decision == DecisionType.DENY:
            self._state.claims_denied += 1

            if is_fraud:
                self._state.true_positives += 1
                self._state.fraud_amount_caught += claim_amount
                reward = claim_amount * reward_cfg.fraud_caught_reward_rate
            else:
                self._state.false_positives += 1
                self._state.false_positive_cost += reward_cfg.false_denial_penalty
                reward = -reward_cfg.false_denial_penalty

        elif decision == DecisionType.REQUEST_INFO:
            self._state.claims_pending_info += 1
            self._state.investigation_cost += reward_cfg.review_cost / 2

            if is_fraud:
                # 50/50 outcome with seeded RNG
                if self._rng.random() < 0.5:
                    self._state.true_positives += 1
                    self._state.fraud_amount_caught += claim_amount
                    reward = (
                        claim_amount * reward_cfg.fraud_caught_reward_rate * 0.5
                        - reward_cfg.review_cost / 2
                    )
                else:
                    self._state.false_negatives += 1
                    self._state.fraud_amount_missed += claim_amount
                    reward = (
                        -claim_amount * reward_cfg.fraud_missed_penalty_rate * 0.5
                        - reward_cfg.review_cost / 2
                    )
            else:
                self._state.true_negatives += 1
                reward = -reward_cfg.unnecessary_info_request_penalty

        return reward

    def _score_rationale(
        self,
        action: ClaimAction,
        claim: ClaimObservation,
        is_fraud: bool,
    ) -> float:
        """
        Score the quality of the LLM's rationale.

        Scoring criteria:
        - Has rationale at all (+0.2)
        - Rationale length reasonable (20-200 words) (+0.2)
        - Mentions relevant fraud indicators (+0.3)
        - Coherent reasoning (+0.3)

        Returns score between 0 and 1.
        """
        if not action.has_rationale():
            return 0.0

        rationale = action.parsed_rationale or ""
        score = 0.2  # Base score for having rationale

        # Length check (roughly 20-200 words)
        word_count = len(rationale.split())
        if 20 <= word_count <= 200:
            score += 0.2
        elif word_count > 0:
            score += 0.1

        # Check for relevant fraud indicator mentions
        fraud_keywords = [
            "upcoding",
            "unbundling",
            "phantom",
            "billing",
            "unusual",
            "anomaly",
            "pattern",
            "deviation",
            "suspicious",
            "high",
            "amount",
            "provider",
            "frequency",
            "duplicate",
            "claim",
        ]
        keyword_matches = sum(1 for kw in fraud_keywords if kw.lower() in rationale.lower())
        score += min(0.3, keyword_matches * 0.05)

        # Check for data point mentions (numbers, percentages)
        has_numbers = any(char.isdigit() for char in rationale)
        has_percentages = "%" in rationale
        if has_numbers or has_percentages:
            score += 0.2

        return min(1.0, score)

    def _score_evidence(self, action: ClaimAction, claim: ClaimObservation) -> float:
        """
        Score the quality of evidence citations.

        Scoring criteria:
        - Has evidence (+0.3)
        - Evidence contains claim-specific data (+0.4)
        - Multiple evidence points (+0.3)

        Returns score between 0 and 1.
        """
        if not action.has_evidence():
            return 0.0

        evidence = action.parsed_evidence or []
        score = 0.3  # Base score for having evidence

        # Check for claim-specific data references
        claim_data_refs = [
            str(claim.claim_amount),
            claim.claim_id,
            claim.provider_profile.get("provider_id", ""),
            claim.member_profile.get("member_id", ""),
        ]

        evidence_text = " ".join(evidence).lower()
        refs_found = sum(1 for ref in claim_data_refs if ref.lower() in evidence_text)
        score += min(0.4, refs_found * 0.1)

        # Bonus for multiple evidence points
        if len(evidence) >= 3:
            score += 0.3
        elif len(evidence) >= 2:
            score += 0.2

        return min(1.0, score)

    def _calculate_efficiency(
        self,
        decision: DecisionType,
        is_fraud: bool,
        claim_amount: float,
        reward_cfg: RewardConfig,
    ) -> float:
        """
        Calculate efficiency reward.

        Rewards choosing the most cost-effective action:
        - APPROVE for legitimate claims (most efficient)
        - FLAG_REVIEW for moderate-risk fraud (balanced)
        - INVESTIGATE for high-value fraud (justified cost)
        - DENY for clear fraud (efficient, but risky)

        Returns normalized score between -1 and 1.
        """
        if is_fraud:
            # For fraud, more aggressive actions are more efficient
            if decision == DecisionType.DENY:
                return 0.5  # Most cost-efficient for fraud
            elif decision == DecisionType.INVESTIGATE:
                return 0.2 if claim_amount > 1000 else -0.1  # Worth it for high-value
            elif decision == DecisionType.FLAG_REVIEW:
                return 0.1
            elif decision == DecisionType.REQUEST_INFO:
                return -0.2  # Delays detection
            else:  # APPROVE
                return -1.0  # Worst - let fraud through
        else:
            # For legitimate claims, less aggressive = more efficient
            if decision == DecisionType.APPROVE:
                return 0.5  # Most efficient
            elif decision == DecisionType.REQUEST_INFO:
                return -0.1  # Slight delay
            elif decision == DecisionType.FLAG_REVIEW:
                return -0.3  # Unnecessary cost
            elif decision == DecisionType.INVESTIGATE:
                return -0.5  # Wasted resources
            else:  # DENY
                return -1.0  # Very bad - denied legitimate claim

    # ------------------------------------------------------------------
    # Enhanced environment helpers: risk scoring, decision recording
    # ------------------------------------------------------------------

    def _calculate_risk_score(self, claim: ClaimObservation) -> float:
        """Compute a per-claim risk score in [0, 1] from claim features.

        Combines amount z-score, provider risk score, and member risk score
        into a single normalised value.  Used for prompt rendering and
        the risk-awareness reward component.

        Args:
            claim: The current claim observation.

        Returns:
            Float in [0.0, 1.0].
        """
        features = claim.claim_features
        if not features:
            return 0.0

        # Weighted combination of available indicators
        amount_z = abs(features.get("amount_zscore", 0.0))
        provider_risk = features.get("provider_risk_score", 0.0)
        member_risk = features.get("member_risk_score", 0.0)

        # Normalise each into roughly [0, 1] and combine
        norm_amount = min(1.0, amount_z / 3.0)  # z>3 → maxed
        norm_provider = min(1.0, max(0.0, provider_risk))
        norm_member = min(1.0, max(0.0, member_risk))

        score = 0.5 * norm_provider + 0.3 * norm_amount + 0.2 * norm_member
        return min(1.0, max(0.0, score))

    def _record_decision(
        self,
        decision: DecisionType,
        claim: ClaimObservation,
        is_fraud: bool,
        outcome: str,
    ) -> None:
        """Append a decision record to state history and update memory.

        Called at the end of ``_process_action`` so the record is available
        for the *next* step's prompt.

        Args:
            decision: The (possibly downgraded) decision type.
            claim: The current claim observation.
            is_fraud: Ground truth for the claim.
            outcome: Human-readable outcome string for prompt display.
        """
        assert self._state is not None

        provider_id = claim.provider_profile.get("provider_id", "???")

        # Append to decision history
        self._state.decision_history.append(
            {
                "step": self._step_count,
                "provider_id": provider_id,
                "amount": claim.claim_amount,
                "decision": decision.value,
                "outcome": outcome,
                "is_fraud": is_fraud,
            }
        )

        # If INVESTIGATE: record result in investigation memory
        if decision == DecisionType.INVESTIGATE:
            inv = InvestigationResult(
                provider_id=provider_id,
                is_fraud=is_fraud,
                step_investigated=self._step_count,
                base_confidence=1.0,
            )
            # Store as dict for Pydantic serialisation
            self._state.investigation_memory[provider_id] = {
                "provider_id": inv.provider_id,
                "is_fraud": inv.is_fraud,
                "step_investigated": inv.step_investigated,
                "base_confidence": inv.base_confidence,
            }

    def _get_episode_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the completed episode."""
        return {
            "episode_id": self._episode_id,
            "total_claims": self._state.total_claims_processed,
            "fraud_detection": {
                "true_positives": self._state.true_positives,
                "false_positives": self._state.false_positives,
                "true_negatives": self._state.true_negatives,
                "false_negatives": self._state.false_negatives,
                "precision": self._state.precision,
                "recall": self._state.recall,
                "f1_score": self._state.f1_score,
            },
            "financial": {
                "total_claim_value": self._state.total_claim_value_processed,
                "fraud_caught": self._state.fraud_amount_caught,
                "fraud_missed": self._state.fraud_amount_missed,
                "investigation_cost": self._state.investigation_cost,
                "false_positive_cost": self._state.false_positive_cost,
                "net_savings": self._state.net_savings,
            },
            "actions": {
                "approved": self._state.claims_approved,
                "flagged": self._state.claims_flagged,
                "investigated": self._state.claims_investigated,
                "denied": self._state.claims_denied,
                "pending_info": self._state.claims_pending_info,
            },
            "llm_quality": {
                "valid_response_rate": self._state.response_validity_rate,
                "avg_rationale_quality": self._state.avg_rationale_quality,
            },
            "enhanced": {
                "budget_used": (self._state.investigation_budget - self._state.budget_remaining),
                "budget_total": self._state.investigation_budget,
                "providers_investigated": len(self._state.investigation_memory),
                "decisions_recorded": len(self._state.decision_history),
            },
            "cumulative_reward": self._state.cumulative_reward,
        }

    @property
    def state(self) -> ClaimState:
        """
        Get the current episode state.

        Returns:
            ClaimState with episode tracking information
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    @property
    def last_reward_components(self) -> Optional[RewardComponents]:
        """Get the reward breakdown from the last step."""
        return self._last_reward_components

    @property
    def action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        return {
            "type": "text_generation",
            "description": "LLM generates structured text with decision, rationale, evidence",
            "decisions": [e.value for e in DecisionType],
            "required_sections": ["Decision", "Rationale", "Evidence"],
            "optional_sections": ["Recommendation"],
            "example": """Decision: FLAG_REVIEW
Rationale: This claim shows patterns consistent with upcoding. The billed amount of $5,000 is 3 standard deviations above the provider's typical claim amount of $500.
Evidence: Provider avg_claim_amount=$500, current claim=$5000, procedure code 99215 unusual for specialty.
Recommendation: Compare with peer providers in same specialty.""",
        }

    @property
    def observation_space_info(self) -> Dict[str, Any]:
        """Get information about the observation space."""
        return {
            "type": "text_prompt",
            "description": "Structured claim data formatted as LLM prompt",
            "sections": [
                "Claim Details",
                "Provider Profile",
                "Member Profile",
                "Risk Indicators",
                "Task Instructions",
            ],
            "feature_vector_dim": 12 + 5,  # For classical RL compatibility
        }

    def render(self, mode: str = "text") -> Optional[str]:
        """
        Render the current environment state.

        Args:
            mode: Rendering mode ("text" or "dict")

        Returns:
            String representation if mode="text", None otherwise
        """
        if self._state is None:
            return "Environment not initialized"

        if mode == "text":
            lines = [
                f"=== Claims Fraud Detection Environment ===",
                f"Episode: {self._episode_id}",
                f"Step: {self._step_count} / {self.config.claims_per_episode}",
                f"",
                f"--- Performance ---",
                f"Precision: {self._state.precision:.2%}",
                f"Recall: {self._state.recall:.2%}",
                f"F1 Score: {self._state.f1_score:.2%}",
                f"",
                f"--- Financial ---",
                f"Fraud Caught: ${self._state.fraud_amount_caught:,.2f}",
                f"Fraud Missed: ${self._state.fraud_amount_missed:,.2f}",
                f"Investigation Cost: ${self._state.investigation_cost:,.2f}",
                f"Net Savings: ${self._state.net_savings:,.2f}",
                f"",
                f"--- LLM Quality ---",
                f"Valid Response Rate: {self._state.response_validity_rate:.2%}",
                f"Avg Rationale Quality: {self._state.avg_rationale_quality:.2f}",
                f"",
                f"--- Cumulative Reward ---",
                f"{self._state.cumulative_reward:.2f}",
                f"",
                f"--- Budget & Memory ---",
                f"Budget: {self._state.budget_remaining}/{self._state.investigation_budget}",
                f"Providers Investigated: {len(self._state.investigation_memory)}",
                f"Decisions Recorded: {len(self._state.decision_history)}",
            ]
            return "\n".join(lines)

        return None
