"""
Data models for the Healthcare Claims Fraud Detection Environment.

Refactored to use OpenEnv types (Pydantic-based Action, Observation, State)
and LLM-native text generation action space.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
import re



@dataclass
class InvestigationResult:
    """Result of a provider investigation, with time-decaying confidence.

    Stored in ClaimState.investigation_memory keyed by provider_id.
    Confidence decays exponentially so stale investigations carry less weight.

    Attributes:
        provider_id: The investigated provider's identifier.
        is_fraud: Whether the investigation found fraud.
        step_investigated: The episode step when the investigation occurred.
        base_confidence: Initial confidence level (1.0 = certain).
    """

    provider_id: str
    is_fraud: bool
    step_investigated: int
    base_confidence: float = 1.0

    def effective_confidence(self, current_step: int, halflife: int = 20) -> float:
        """Compute decayed confidence at *current_step*.

        Uses exponential decay: ``base_confidence * decay_rate ^ turns_elapsed``
        where ``decay_rate = 0.5 ^ (1 / halflife)``.

        Args:
            current_step: The current episode step number.
            halflife: Number of turns for confidence to halve (default 20).

        Returns:
            Decayed confidence in [0, base_confidence].
        """
        if halflife <= 0:
            return 0.0
        decay_rate = 0.5 ** (1 / halflife)
        turns_elapsed = max(0, current_step - self.step_investigated)
        return self.base_confidence * (decay_rate**turns_elapsed)


class DecisionType(str, Enum):
    """Available decisions for claims processing."""

    APPROVE = "APPROVE"
    """Approve the claim for payment without further review."""

    FLAG_REVIEW = "FLAG_REVIEW"
    """Flag for manual review by a claims examiner."""

    INVESTIGATE = "INVESTIGATE"
    """Initiate a full fraud investigation (higher cost, higher accuracy)."""

    DENY = "DENY"
    """Deny the claim outright (risky - may trigger appeals)."""

    REQUEST_INFO = "REQUEST_INFO"
    """Request additional documentation before deciding."""


class ClaimAction(BaseModel):
    """
    LLM-generated action for processing a healthcare claim.

    The LLM must generate a structured text response containing:
    - A decision (APPROVE, FLAG_REVIEW, INVESTIGATE, DENY, REQUEST_INFO)
    - A rationale explaining why this decision was made
    - Evidence citations from the claim data
    - Optional: Recommended follow-up actions

    This design requires LLM text generation capabilities, not just classification.

    Example valid response:
        Decision: FLAG_REVIEW
        Rationale: This claim shows patterns consistent with upcoding. The billed
        amount of $5,000 is 3 standard deviations above the provider's typical
        claim amount of $500.
        Evidence: Provider avg_claim_amount=$500, current claim=$5000,
        procedure code 99215 unusual for specialty.
        Recommendation: Compare with peer providers in same specialty.
    """

    response_text: str = Field(
        ..., description="Full LLM-generated response containing decision, rationale, and evidence"
    )

    # Parsed fields (populated by parser, optional for input)
    parsed_decision: Optional[str] = Field(
        default=None, description="Extracted decision type from response_text"
    )
    parsed_rationale: Optional[str] = Field(
        default=None, description="Extracted rationale from response_text"
    )
    parsed_evidence: Optional[List[str]] = Field(
        default=None, description="Extracted evidence citations from response_text"
    )
    parsed_recommendation: Optional[str] = Field(
        default=None, description="Extracted recommendation from response_text"
    )

    def parse_response(self) -> "ClaimAction":
        """
        Parse the response_text to extract structured components.

        Returns self with parsed fields populated.
        """
        text = self.response_text.upper()

        # Extract decision
        decision = None
        for dt in DecisionType:
            if dt.value in text:
                decision = dt.value
                break

        # Try regex patterns for more precise extraction
        decision_pattern = r"DECISION:\s*(\w+)"
        match = re.search(decision_pattern, text)
        if match:
            potential_decision = match.group(1).upper()
            for dt in DecisionType:
                if potential_decision in dt.value or dt.value in potential_decision:
                    decision = dt.value
                    break

        self.parsed_decision = decision

        # Extract rationale (text after "Rationale:" until next section or end)
        rationale_pattern = r"RATIONALE:\s*(.+?)(?=EVIDENCE:|RECOMMENDATION:|$)"
        match = re.search(rationale_pattern, self.response_text, re.IGNORECASE | re.DOTALL)
        if match:
            self.parsed_rationale = match.group(1).strip()

        # Extract evidence (text after "Evidence:")
        evidence_pattern = r"EVIDENCE:\s*(.+?)(?=RECOMMENDATION:|$)"
        match = re.search(evidence_pattern, self.response_text, re.IGNORECASE | re.DOTALL)
        if match:
            evidence_text = match.group(1).strip()
            # Split by commas or newlines
            self.parsed_evidence = [
                e.strip() for e in re.split(r"[,\n]", evidence_text) if e.strip()
            ]

        # Extract recommendation
        rec_pattern = r"RECOMMENDATION:\s*(.+?)$"
        match = re.search(rec_pattern, self.response_text, re.IGNORECASE | re.DOTALL)
        if match:
            self.parsed_recommendation = match.group(1).strip()

        return self

    def get_decision(self) -> Optional[DecisionType]:
        """Get the parsed decision as a DecisionType enum."""
        if self.parsed_decision is None:
            self.parse_response()

        if self.parsed_decision:
            try:
                return DecisionType(self.parsed_decision)
            except ValueError:
                return None
        return None

    def has_valid_decision(self) -> bool:
        """Check if the response contains a valid decision."""
        return self.get_decision() is not None

    def has_rationale(self) -> bool:
        """Check if the response contains a rationale."""
        if self.parsed_rationale is None:
            self.parse_response()
        return bool(self.parsed_rationale and len(self.parsed_rationale) > 10)

    def has_evidence(self) -> bool:
        """Check if the response contains evidence citations."""
        if self.parsed_evidence is None:
            self.parse_response()
        return bool(self.parsed_evidence and len(self.parsed_evidence) > 0)


class ProviderProfile(BaseModel):
    """
    Summary statistics about a healthcare provider's billing patterns.

    These features help detect anomalous provider behavior.
    Uses Action as base for Pydantic compatibility.
    """

    provider_id: str = Field(..., description="Unique provider identifier")
    specialty: str = Field(..., description="Provider medical specialty")
    total_claims_30d: int = Field(..., description="Total claims in last 30 days")
    total_amount_30d: float = Field(..., description="Total billed amount in last 30 days")
    avg_claim_amount: float = Field(..., description="Average claim amount")
    claim_denial_rate: float = Field(..., ge=0, le=1, description="Rate of denied claims")
    fraud_flag_rate: float = Field(..., ge=0, le=1, description="Rate of fraud flags")
    unique_patients_30d: int = Field(..., description="Unique patients in last 30 days")
    unique_procedures_30d: int = Field(..., description="Unique procedures in last 30 days")
    weekend_claim_rate: float = Field(..., ge=0, le=1, description="Rate of weekend claims")
    high_cost_procedure_rate: float = Field(
        ..., ge=0, le=1, description="Rate of high-cost procedures"
    )


class MemberProfile(BaseModel):
    """
    Summary statistics about a member's claim history.

    These features help detect anomalous member behavior or identity theft.
    Uses Action as base for Pydantic compatibility.
    """

    member_id: str = Field(..., description="Unique member identifier")
    age: int = Field(..., ge=0, le=120, description="Member age")
    gender: str = Field(..., description="Member gender")
    chronic_condition_count: int = Field(..., ge=0, description="Number of chronic conditions")
    total_claims_90d: int = Field(..., ge=0, description="Total claims in last 90 days")
    total_amount_90d: float = Field(..., ge=0, description="Total claim amount in last 90 days")
    unique_providers_90d: int = Field(..., ge=0, description="Unique providers in last 90 days")
    er_visit_count_90d: int = Field(..., ge=0, description="ER visits in last 90 days")
    prescription_count_90d: int = Field(..., ge=0, description="Prescriptions in last 90 days")
    avg_days_between_claims: float = Field(..., ge=0, description="Average days between claims")


class ClaimObservation(BaseModel):
    """
    Observation representing a single healthcare claim to be processed.

    This is the state visible to the RL agent at each step. The observation
    includes all information an LLM would need to make an informed decision
    about whether this claim is potentially fraudulent.

    Inherits from OpenEnv Observation which provides:
    - done: bool (episode terminated)
    - reward: float | None (reward from last action)
    - metadata: Dict[str, Any] (additional metadata)
    """

    # Fields previously inherited from OpenEnv Observation base class
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: Optional[float] = Field(default=None, description="Reward from the last action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")

    # Pydantic config: allow mutation for environment updates
    model_config = {"arbitrary_types_allowed": True}

    # Claim identification
    claim_id: str = Field(..., description="Unique identifier for the claim")

    # Claim details
    claim_amount: float = Field(..., ge=0, description="Total billed amount in USD")
    procedure_codes: List[str] = Field(..., description="List of CPT/HCPCS procedure codes")
    diagnosis_codes: List[str] = Field(..., description="List of ICD-10 diagnosis codes")
    place_of_service: str = Field(..., description="Where service was rendered")
    service_date: str = Field(..., description="Date of service (YYYY-MM-DD)")
    submission_date: str = Field(..., description="Date claim was submitted")

    # Provider and member profiles (as dicts for flexibility)
    provider_profile: Dict[str, Any] = Field(..., description="Provider billing statistics")
    member_profile: Dict[str, Any] = Field(..., description="Member claim history")

    # Derived features for fraud detection
    claim_features: Dict[str, float] = Field(
        default_factory=dict, description="Computed features for fraud detection"
    )

    # Prompt for LLM (pre-formatted observation as text)
    prompt: Optional[str] = Field(
        default=None, description="Pre-formatted prompt for LLM consumption"
    )

    def to_prompt(
        self,
        state: Optional["ClaimState"] = None,
        history_window: int = 20,
        memory_halflife: int = 20,
    ) -> str:
        """Convert observation to a natural language prompt for LLM.

        This is the key LLM-native transformation — presenting structured
        data as a text prompt that the LLM can reason about.

        When *state* is provided the prompt includes episode context:
        budget status, investigation memory with decayed confidence,
        windowed decision history, and per-claim risk assessment.

        Args:
            state: Current ``ClaimState`` for episode context.  ``None``
                falls back to the original single-claim prompt (backward
                compatible).
            history_window: Maximum number of recent decisions to include
                in the prompt (default 20).
            memory_halflife: Halflife in turns for memory confidence decay
                (default 20).

        Returns:
            Formatted prompt string ready for LLM consumption.
        """
        provider = self.provider_profile
        member = self.member_profile
        features = self.claim_features

        parts: List[str] = []

        # --- Episode state header (only when state is provided) ---
        if state is not None:
            step_display = state.step_count + 1  # 1-indexed for readability
            total_claims = state.remaining_claims_in_batch + state.total_claims_processed
            if total_claims <= 0:
                total_claims = step_display  # fallback during first step

            risk_label = (
                "HIGH"
                if state.risk_score > 0.7
                else "Moderate"
                if state.risk_score > 0.4
                else "Low"
            )

            parts.append(
                f"## Episode State\n"
                f"- Step: {step_display}/{total_claims} "
                f"| Budget: {state.budget_remaining}/{state.investigation_budget} remaining "
                f"| Risk Level: {risk_label}\n"
            )

            # --- Windowed decision history ---
            if state.decision_history:
                recent = state.decision_history[-history_window:]
                history_lines = [
                    "## Your Recent Decisions (last "
                    f"{min(len(state.decision_history), history_window)}):"
                ]
                for entry in recent:
                    outcome = entry.get("outcome", "")
                    outcome_str = f" ({outcome})" if outcome else ""
                    history_lines.append(
                        f"- Step {entry.get('step', '?')}: "
                        f"Claim ${entry.get('amount', 0):,.2f} from "
                        f"{entry.get('provider_id', '???')} → "
                        f"{entry.get('decision', '???')}{outcome_str}"
                    )
                parts.append("\n".join(history_lines) + "\n")

            # --- Active investigation memory ---
            if state.investigation_memory:
                memory_lines = ["## Active Memory:"]
                current_step = state.step_count
                for pid, mem_data in state.investigation_memory.items():
                    # mem_data is a dict serialisation of InvestigationResult
                    inv = InvestigationResult(
                        provider_id=mem_data["provider_id"],
                        is_fraud=mem_data["is_fraud"],
                        step_investigated=mem_data["step_investigated"],
                        base_confidence=mem_data.get("base_confidence", 1.0),
                    )
                    conf = inv.effective_confidence(current_step, halflife=memory_halflife)
                    if conf < 0.05:
                        continue  # too stale to show
                    turns_ago = current_step - inv.step_investigated
                    finding = "FRAUD" if inv.is_fraud else "LEGIT"
                    memory_lines.append(
                        f"- ⚠ {pid}: Investigated {turns_ago} turns ago "
                        f"(Confidence: {conf:.0%}). Finding: {finding}"
                    )
                if len(memory_lines) > 1:
                    parts.append("\n".join(memory_lines) + "\n")

        # --- Current claim details (always shown) ---
        step_label = ""
        if state is not None:
            step_label = f" (Step {state.step_count + 1})"

        parts.append(f"""## Current Claim{step_label}
- **Claim ID**: {self.claim_id}
- **Billed Amount**: ${self.claim_amount:,.2f}
- **Service Date**: {self.service_date}
- **Submission Date**: {self.submission_date}
- **Place of Service**: {self.place_of_service}
- **Procedure Codes**: {", ".join(self.procedure_codes)}
- **Diagnosis Codes**: {", ".join(self.diagnosis_codes)}

## Provider Profile (ID: {provider.get("provider_id", "Unknown")})
- **Specialty**: {provider.get("specialty", "Unknown")}
- **Claims (30 days)**: {provider.get("total_claims_30d", 0)} claims, ${provider.get("total_amount_30d", 0):,.2f} total
- **Average Claim**: ${provider.get("avg_claim_amount", 0):,.2f}
- **Denial Rate**: {provider.get("claim_denial_rate", 0):.1%}
- **Prior Fraud Flags**: {provider.get("fraud_flag_rate", 0):.1%}
- **Weekend Claims**: {provider.get("weekend_claim_rate", 0):.1%}
- **High-Cost Procedures**: {provider.get("high_cost_procedure_rate", 0):.1%}

## Member Profile (ID: {member.get("member_id", "Unknown")})
- **Age/Gender**: {member.get("age", 0)} / {member.get("gender", "Unknown")}
- **Chronic Conditions**: {member.get("chronic_condition_count", 0)}
- **Claims (90 days)**: {member.get("total_claims_90d", 0)} claims, ${member.get("total_amount_90d", 0):,.2f} total
- **Unique Providers**: {member.get("unique_providers_90d", 0)}
- **ER Visits**: {member.get("er_visit_count_90d", 0)}
- **Days Between Claims**: {member.get("avg_days_between_claims", 0):.1f} avg
""")

        # --- Risk indicators ---
        risk_section = "## Risk Indicators\n"
        if state is not None and state.risk_score > 0.0:
            risk_level = (
                "HIGH"
                if state.risk_score > 0.7
                else "MODERATE"
                if state.risk_score > 0.4
                else "LOW"
            )
            risk_section += f"- **Risk Assessment**: {risk_level} ({state.risk_score:.2f})\n"

        # Check if current provider is in memory
        if state is not None and state.investigation_memory:
            current_pid = provider.get("provider_id", "")
            if current_pid in state.investigation_memory:
                mem = state.investigation_memory[current_pid]
                finding = "FRAUD" if mem.get("is_fraud") else "LEGIT"
                risk_section += f"- **⚠ KNOWN PROVIDER**: Previously investigated → {finding}\n"

        if features:
            for key, value in features.items():
                indicator_name = key.replace("_", " ").title()
                if isinstance(value, float):
                    if value > 1:
                        risk_section += f"- **{indicator_name}**: {value:.2f}\n"
                    else:
                        risk_section += f"- **{indicator_name}**: {value:.1%}\n"
                else:
                    risk_section += f"- **{indicator_name}**: {value}\n"
        parts.append(risk_section)

        # --- Task instructions ---
        parts.append("""## Your Task
Analyze this claim for potential fraud indicators and provide your decision.

Respond with:
1. **Decision**: One of [APPROVE, FLAG_REVIEW, INVESTIGATE, DENY, REQUEST_INFO]
2. **Rationale**: Explain your reasoning (2-4 sentences)
3. **Evidence**: List specific data points that influenced your decision
4. **Recommendation**: Any follow-up actions (optional)

Format your response as:
Decision: [YOUR_DECISION]
Rationale: [YOUR_EXPLANATION]
Evidence: [SPECIFIC_DATA_POINTS]
Recommendation: [FOLLOW_UP_ACTIONS]
""")
        return "\n".join(parts)

    def to_feature_vector(self) -> List[float]:
        """
        Convert observation to a flat feature vector for classical RL algorithms.

        Returns:
            List of floats representing the observation
        """
        features = [
            self.claim_amount,
            len(self.procedure_codes),
            len(self.diagnosis_codes),
            self.provider_profile.get("total_claims_30d", 0),
            self.provider_profile.get("avg_claim_amount", 0),
            self.provider_profile.get("claim_denial_rate", 0),
            self.provider_profile.get("fraud_flag_rate", 0),
            self.provider_profile.get("weekend_claim_rate", 0),
            self.member_profile.get("age", 0),
            self.member_profile.get("chronic_condition_count", 0),
            self.member_profile.get("total_claims_90d", 0),
            self.member_profile.get("unique_providers_90d", 0),
        ]
        # Add claim features
        features.extend(self.claim_features.values())
        return features


class ClaimState(BaseModel):
    """
    Episode state tracking for the fraud detection environment.

    Extends OpenEnv State which provides:
    - episode_id: str | None
    - step_count: int (>= 0)

    Adds healthcare-specific tracking fields.
    """

    # Fields previously inherited from OpenEnv State base class
    episode_id: Optional[str] = Field(default=None, description="Episode identifier")
    step_count: int = Field(default=0, ge=0, description="Current step number in episode")

    # Claim processing counts
    total_claims_processed: int = Field(default=0, ge=0)
    claims_approved: int = Field(default=0, ge=0)
    claims_flagged: int = Field(default=0, ge=0)
    claims_investigated: int = Field(default=0, ge=0)
    claims_denied: int = Field(default=0, ge=0)
    claims_pending_info: int = Field(default=0, ge=0)

    # Fraud detection performance
    true_positives: int = Field(default=0, ge=0, description="Correctly identified fraud")
    false_positives: int = Field(default=0, ge=0, description="Incorrectly flagged legitimate")
    true_negatives: int = Field(default=0, ge=0, description="Correctly approved legitimate")
    false_negatives: int = Field(default=0, ge=0, description="Missed fraud")

    # Financial metrics
    total_claim_value_processed: float = Field(default=0.0, ge=0)
    fraud_amount_caught: float = Field(default=0.0, ge=0)
    fraud_amount_missed: float = Field(default=0.0, ge=0)
    investigation_cost: float = Field(default=0.0, ge=0)
    false_positive_cost: float = Field(default=0.0, ge=0)

    # Cumulative reward
    cumulative_reward: float = Field(default=0.0)

    # Queue status
    remaining_claims_in_batch: int = Field(default=0, ge=0)

    # LLM-specific metrics (new for text generation)
    rationale_quality_sum: float = Field(default=0.0, description="Sum of rationale quality scores")
    evidence_citation_sum: float = Field(default=0.0, description="Sum of evidence citation scores")
    valid_response_count: int = Field(default=0, ge=0, description="Count of valid LLM responses")
    invalid_response_count: int = Field(
        default=0, ge=0, description="Count of invalid/unparseable responses"
    )

    # --- Enhanced environment fields (investigation budget, memory, history) ---

    # Investigation budget: total allowed and remaining this episode
    investigation_budget: int = Field(
        default=15, ge=1, description="Total investigation budget per episode"
    )
    budget_remaining: int = Field(
        default=15, ge=0, description="Remaining investigations this episode"
    )

    # Provider investigation memory: provider_id -> serialised InvestigationResult
    # Stored as Dict[str, Any] for Pydantic JSON compatibility; the environment
    # layer converts to/from InvestigationResult objects.
    investigation_memory: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider investigation results keyed by provider_id",
    )

    # Decision history: list of per-step decision records
    decision_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological list of decision records for the episode",
    )

    # Per-claim risk score (set by environment before prompt generation)
    risk_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Risk score for the current claim"
    )

    @property
    def precision(self) -> float:
        """Calculate precision (of fraud detection)."""
        total_flagged = self.true_positives + self.false_positives
        return self.true_positives / total_flagged if total_flagged > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall (of fraud detection)."""
        total_fraud = self.true_positives + self.false_negatives
        return self.true_positives / total_fraud if total_fraud > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def net_savings(self) -> float:
        """Calculate net financial savings from fraud detection."""
        return (
            self.fraud_amount_caught
            - self.investigation_cost
            - self.false_positive_cost
            - self.fraud_amount_missed
        )

    @property
    def avg_rationale_quality(self) -> float:
        """Average rationale quality score."""
        if self.valid_response_count == 0:
            return 0.0
        return self.rationale_quality_sum / self.valid_response_count

    @property
    def response_validity_rate(self) -> float:
        """Rate of valid (parseable) LLM responses."""
        total = self.valid_response_count + self.invalid_response_count
        return self.valid_response_count / total if total > 0 else 0.0


class RewardComponents(BaseModel):
    """
    Multi-component reward breakdown for LLM-native tasks.

    The reward is composed of multiple factors:
    1. Decision correctness (did the agent make the right call?)
    2. Rationale quality (is the explanation coherent and relevant?)
    3. Evidence citation (did the agent cite specific data points?)
    4. Efficiency (was this the most cost-effective action?)

    This enables training LLMs to not just classify correctly,
    but to explain their reasoning - a key LLM capability.
    """

    # Decision component (0.4 weight default)
    decision_reward: float = Field(default=0.0, description="Reward for correct/incorrect decision")

    # Rationale quality component (0.3 weight default)
    rationale_reward: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Reward for rationale quality (coherence, relevance)",
    )

    # Evidence citation component (0.2 weight default)
    evidence_reward: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Reward for citing relevant evidence from claim data",
    )

    # Efficiency component (0.1 weight default)
    efficiency_reward: float = Field(
        default=0.0, description="Reward for cost-effective action selection"
    )

    # Parse validity penalty
    parse_penalty: float = Field(
        default=0.0, le=0.0, description="Penalty for unparseable responses"
    )

    # Weights
    decision_weight: float = Field(default=0.4)
    rationale_weight: float = Field(default=0.3)
    evidence_weight: float = Field(default=0.2)
    efficiency_weight: float = Field(default=0.1)

    @property
    def total_reward(self) -> float:
        """Calculate weighted total reward."""
        weighted = (
            self.decision_reward * self.decision_weight
            + self.rationale_reward * self.rationale_weight
            + self.evidence_reward * self.evidence_weight
            + self.efficiency_reward * self.efficiency_weight
            + self.parse_penalty
        )
        return weighted

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "decision_reward": self.decision_reward,
            "rationale_reward": self.rationale_reward,
            "evidence_reward": self.evidence_reward,
            "efficiency_reward": self.efficiency_reward,
            "parse_penalty": self.parse_penalty,
            "total_reward": self.total_reward,
        }


class RewardConfig(BaseModel):
    """
    Configuration for reward function parameters.

    These can be tuned to adjust the agent's behavior.
    Uses Action as base for Pydantic compatibility.
    """

    # Reward for catching fraud (per dollar of fraud caught)
    fraud_caught_reward_rate: float = Field(default=0.1)

    # Penalty for missing fraud (per dollar of missed fraud)
    fraud_missed_penalty_rate: float = Field(default=0.2)

    # Cost of investigation (flat fee per investigation)
    investigation_cost: float = Field(default=100.0)

    # Cost of flagging for review (lower than full investigation)
    review_cost: float = Field(default=25.0)

    # Penalty for false positive (incorrectly flagging legitimate claim)
    false_positive_penalty: float = Field(default=50.0)

    # Penalty for denied legitimate claim (high - triggers appeals)
    false_denial_penalty: float = Field(default=200.0)

    # Small reward for efficient processing (approved legitimate claim quickly)
    efficient_approval_reward: float = Field(default=1.0)

    # Penalty for requesting info unnecessarily
    unnecessary_info_request_penalty: float = Field(default=10.0)

    # Penalty for unparseable LLM response
    invalid_response_penalty: float = Field(default=5.0)

    # Bonus for high-quality rationale
    rationale_quality_bonus: float = Field(default=2.0)

    # Bonus for citing evidence
    evidence_citation_bonus: float = Field(default=1.0)


# Type aliases for clarity
StepResult = ClaimObservation  # OpenEnv uses Observation as step result
