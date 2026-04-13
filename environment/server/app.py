"""
FastAPI Server for the Healthcare Claims Fraud Detection Environment.

This server exposes the environment via HTTP endpoints following
the OpenEnv specification, plus Green Agent A2A endpoints for
AgentBeats compatibility.

Run with (from repo root):
    uvicorn environment.server.app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime

from .environment import ClaimsFraudEnvironment, EnvironmentConfig
from ..models import ClaimAction, DecisionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Healthcare Claims Fraud Detection Environment",
    description="""
    An OpenEnv-compatible RL environment for sequential fraud detection
    in healthcare insurance claims.
    
    ## LLM-Native Design
    
    This environment is designed for LLM agents. The agent receives claims
    as text prompts and must generate structured text responses containing:
    
    - **Decision**: APPROVE, FLAG_REVIEW, INVESTIGATE, DENY, or REQUEST_INFO
    - **Rationale**: Explanation of the reasoning (2-4 sentences)
    - **Evidence**: Specific data points that influenced the decision
    - **Recommendation**: Optional follow-up actions
    
    ## Multi-Component Rewards
    
    - **Decision correctness** (40%): Did you make the right call?
    - **Rationale quality** (30%): Is your explanation coherent and relevant?
    - **Evidence citation** (20%): Did you cite specific claim data?
    - **Efficiency** (10%): Was this the most cost-effective action?
    
    ## A2A Protocol Support
    
    This environment implements the Agent-to-Agent protocol for
    AgentBeats/Green Agent compatibility:
    
    - `/a2a/agent-card`: Capability discovery
    - `/a2a/assess`: Task assessment and scoring
    """,
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env: Optional[ClaimsFraudEnvironment] = None


# =============================================================================
# Pydantic models for request/response
# =============================================================================


class LLMActionRequest(BaseModel):
    """Request model for LLM-generated step action."""

    response_text: str = Field(
        ...,
        description="Full LLM-generated response containing decision, rationale, evidence",
        json_schema_extra={
            "example": """Decision: FLAG_REVIEW
Rationale: This claim shows patterns consistent with upcoding. The billed amount is elevated compared to provider's typical claims.
Evidence: Provider avg_claim_amount=$500, current claim=$5000, procedure code 99215 unusual for specialty.
Recommendation: Compare with peer providers."""
        },
    )


class LegacyActionRequest(BaseModel):
    """Legacy request model for discrete actions (backward compatibility)."""

    action_type: str
    confidence: Optional[float] = None
    notes: Optional[str] = None


class ObservationResponse(BaseModel):
    """Response model for observations."""

    claim_id: str
    claim_amount: float
    procedure_codes: List[str]
    diagnosis_codes: List[str]
    place_of_service: str
    service_date: str
    submission_date: str
    provider_profile: Dict[str, Any]
    member_profile: Dict[str, Any]
    claim_features: Dict[str, float]
    prompt: Optional[str] = Field(None, description="Pre-formatted LLM prompt")
    done: bool = False
    reward: Optional[float] = None


class StepResponse(BaseModel):
    """Response model for step results."""

    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    """Response model for episode state."""

    episode_id: Optional[str] = None
    step_count: int
    total_claims_processed: int
    claims_approved: int
    claims_flagged: int
    claims_investigated: int
    claims_denied: int
    claims_pending_info: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_claim_value_processed: float
    fraud_amount_caught: float
    fraud_amount_missed: float
    investigation_cost: float
    false_positive_cost: float
    cumulative_reward: float
    remaining_claims_in_batch: int
    # LLM quality metrics
    rationale_quality_sum: float = 0.0
    evidence_citation_sum: float = 0.0
    valid_response_count: int = 0
    invalid_response_count: int = 0


class ConfigRequest(BaseModel):
    """Request model for environment configuration.

    All fields are optional — unset fields use environment defaults.
    Exposed via POST /reset so callers can reproduce budget and memory ablations
    without needing direct Python access (e.g. from the HF Space or the green agent).
    """

    claims_per_episode: int = 100
    fraud_rate: float = 0.05
    seed: Optional[int] = None
    # Budget / memory fields for ablation reproducibility
    investigation_budget: int = 15
    memory_decay_halflife: int = 20


# =============================================================================
# A2A Protocol Models (Green Agent)
# =============================================================================


class A2ACapability(BaseModel):
    """A2A capability descriptor."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class A2AAgentCard(BaseModel):
    """Agent capability card for A2A protocol."""

    name: str
    description: str
    version: str
    capabilities: List[A2ACapability]
    endpoints: Dict[str, str]
    metadata: Dict[str, Any]


class A2AAssessRequest(BaseModel):
    """Request for A2A assessment."""

    task_id: str
    agent_response: str
    ground_truth: Optional[Dict[str, Any]] = None


class A2AAssessResponse(BaseModel):
    """Response from A2A assessment."""

    task_id: str
    score: float
    breakdown: Dict[str, float]
    feedback: str
    metadata: Dict[str, Any]


# =============================================================================
# Standard OpenEnv Endpoints
# =============================================================================


@app.on_event("startup")
async def startup():
    """Initialize environment on server startup."""
    global env
    logger.info("Initializing Claims Fraud Detection Environment (LLM-native)...")
    env = ClaimsFraudEnvironment()
    logger.info("Environment initialized successfully")


# =============================================================================
# OpenEnv Manifest Endpoint
# =============================================================================

@app.get("/.well-known/openenv.yaml", response_class=PlainTextResponse)
async def openenv_manifest():
    """
    OpenEnv manifest — required by the OpenEnv specification.
    Discoverable at /.well-known/openenv.yaml so the hub can index this Space.
    """
    # Resolve path relative to this file so it works both locally and in Docker
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "openenv.yaml")
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.exists(yaml_path):
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    with open(yaml_path, "r") as f:
        content = f.read()
    return PlainTextResponse(content=content, media_type="application/yaml")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns status of the environment server.
    """
    return {
        "status": "healthy",
        "environment": "healthcare-claims-fraud",
        "version": "0.2.0",
        "features": ["llm-native", "multi-component-reward", "a2a-compatible"],
    }


@app.post("/reset", response_model=StepResponse)
async def reset(config: Optional[ConfigRequest] = None):
    """
    Reset the environment and start a new episode.

    Optionally accepts configuration parameters to customize the episode.
    Returns the initial observation with LLM prompt.
    """
    global env

    if config:
        env_config = EnvironmentConfig(
            claims_per_episode=config.claims_per_episode,
            fraud_rate=config.fraud_rate,
            seed=config.seed,
            investigation_budget=config.investigation_budget,
            memory_decay_halflife=config.memory_decay_halflife,
        )
        env = ClaimsFraudEnvironment(env_config)

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    obs = env.reset()

    return {
        "observation": obs.model_dump(),
        "reward": obs.reward or 0.0,
        "done": obs.done,
        "info": obs.metadata or {},
    }


@app.post("/step", response_model=StepResponse)
async def step(action: LLMActionRequest):
    """
    Execute an LLM-generated action in the environment.

    The action should contain the full LLM response text with:
    - Decision (APPROVE, FLAG_REVIEW, INVESTIGATE, DENY, REQUEST_INFO)
    - Rationale (explanation)
    - Evidence (data citations)
    - Recommendation (optional)
    """
    global env

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        claim_action = ClaimAction(response_text=action.response_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")

    try:
        obs = env.step(claim_action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": obs.reward or 0.0,
        "done": obs.done,
        "info": obs.metadata or {},
    }


@app.post("/step_legacy", response_model=StepResponse)
async def step_legacy(action: LegacyActionRequest):
    """
    Execute a discrete action (legacy/backward compatibility).

    Converts discrete action to LLM format internally.
    """
    global env

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    # Convert legacy action to LLM format
    response_text = f"""Decision: {action.action_type.upper()}
Rationale: {action.notes or "No rationale provided."}
Evidence: N/A
"""

    try:
        claim_action = ClaimAction(response_text=response_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")

    try:
        obs = env.step(claim_action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": obs.reward or 0.0,
        "done": obs.done,
        "info": obs.metadata or {},
    }


@app.get("/state", response_model=StateResponse)
async def get_state():
    """
    Get the current episode state.

    Returns tracking information including LLM quality metrics.
    """
    global env

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        state = env.state
        return state.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/action_space")
async def get_action_space():
    """
    Get information about the LLM action space.

    Returns required response format and example.
    """
    global env

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    return env.action_space_info


@app.get("/observation_space")
async def get_observation_space():
    """
    Get information about the observation space.

    Returns structure of observations and prompt format.
    """
    global env

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    return env.observation_space_info


@app.get("/render")
async def render(mode: str = "text"):
    """
    Render the current environment state.

    Args:
        mode: Rendering mode ("text" or "dict")
    """
    global env

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        result = env.render(mode)
        return {"render": result}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """
    Get simulator statistics.
    """
    global env

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    return env.simulator.get_statistics()


# =============================================================================
# A2A Protocol Endpoints (Green Agent)
# =============================================================================


@app.get("/a2a/agent-card", response_model=A2AAgentCard)
async def get_agent_card():
    """
    A2A Agent Card - Capability discovery endpoint.

    Returns information about this environment's capabilities
    for AgentBeats/Green Agent integration.
    """
    return A2AAgentCard(
        name="Healthcare Claims Fraud Detection Environment",
        description="RL environment for training LLM agents to detect healthcare insurance fraud",
        version="0.2.0",
        capabilities=[
            A2ACapability(
                name="fraud_detection",
                description="Analyze healthcare claims and determine fraud likelihood",
                input_schema={
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "claim_amount": {"type": "number"},
                        "procedure_codes": {"type": "array", "items": {"type": "string"}},
                        "diagnosis_codes": {"type": "array", "items": {"type": "string"}},
                        "provider_profile": {"type": "object"},
                        "member_profile": {"type": "object"},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": [
                                "APPROVE",
                                "FLAG_REVIEW",
                                "INVESTIGATE",
                                "DENY",
                                "REQUEST_INFO",
                            ],
                        },
                        "rationale": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "recommendation": {"type": "string"},
                    },
                },
            )
        ],
        endpoints={
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "assess": "/a2a/assess",
        },
        metadata={
            "domain": "healthcare",
            "task_type": "fraud_detection",
            "action_type": "text_generation",
            "reward_type": "multi_component",
            "openenv_compatible": True,
            "trl_compatible": True,
        },
    )


@app.post("/a2a/assess", response_model=A2AAssessResponse)
async def assess_response(request: A2AAssessRequest):
    """
    A2A Assessment endpoint - Score an agent's response.

    Used by AgentBeats to evaluate agent performance on tasks.
    """
    global env

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    # Parse the agent response
    action = ClaimAction(response_text=request.agent_response)
    action.parse_response()

    # Get current claim for scoring
    current_obs = env._current_observation
    is_fraud = env._current_is_fraud if hasattr(env, "_current_is_fraud") else None

    # Score the response
    has_valid_decision = action.has_valid_decision()
    has_rationale = action.has_rationale()
    has_evidence = action.has_evidence()

    # Calculate component scores
    parse_score = 1.0 if has_valid_decision else 0.0
    rationale_score = 0.0
    evidence_score = 0.0

    if current_obs and has_valid_decision:
        rationale_score = env._score_rationale(action, current_obs, is_fraud or False)
        evidence_score = env._score_evidence(action, current_obs)

    # Decision correctness (if ground truth provided)
    decision_score = 0.0
    if request.ground_truth and has_valid_decision:
        expected_decision = request.ground_truth.get("decision", "").upper()
        actual_decision = action.parsed_decision or ""
        if expected_decision == actual_decision:
            decision_score = 1.0
        elif expected_decision in ["FLAG_REVIEW", "INVESTIGATE"] and actual_decision in [
            "FLAG_REVIEW",
            "INVESTIGATE",
        ]:
            decision_score = 0.7  # Partial credit for cautious decisions

    # Overall score
    overall_score = (
        0.3 * parse_score + 0.3 * decision_score + 0.2 * rationale_score + 0.2 * evidence_score
    )

    # Generate feedback
    feedback_parts = []
    if not has_valid_decision:
        feedback_parts.append("Response could not be parsed. Ensure format: 'Decision: [DECISION]'")
    if not has_rationale:
        feedback_parts.append("Missing or insufficient rationale. Explain your reasoning.")
    if not has_evidence:
        feedback_parts.append("No evidence cited. Reference specific claim data.")
    if decision_score < 1.0 and request.ground_truth:
        feedback_parts.append(
            f"Decision did not match expected: {request.ground_truth.get('decision')}"
        )

    feedback = (
        " ".join(feedback_parts)
        if feedback_parts
        else "Good response with clear decision, rationale, and evidence."
    )

    return A2AAssessResponse(
        task_id=request.task_id,
        score=overall_score,
        breakdown={
            "parse_validity": parse_score,
            "decision_correctness": decision_score,
            "rationale_quality": rationale_score,
            "evidence_citation": evidence_score,
        },
        feedback=feedback,
        metadata={
            "parsed_decision": action.parsed_decision,
            "response_length": len(request.agent_response),
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.post("/a2a/generate_task")
async def generate_task():
    """
    Generate a new task for agent assessment.

    Resets environment and returns the first claim as a task.
    """
    global env

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    obs = env.reset()

    # NOTE: ground_truth is intentionally excluded from this response.
    # Exposing is_fraud here would allow any caller to trivially cheat the
    # environment by reading the label before deciding. Scoring is done via
    # /a2a/assess after the agent submits its response.
    return {
        "task_id": obs.claim_id,
        "prompt": obs.prompt,
        "claim_data": {
            "claim_id": obs.claim_id,
            "claim_amount": obs.claim_amount,
            "procedure_codes": obs.procedure_codes,
            "diagnosis_codes": obs.diagnosis_codes,
            "provider_profile": obs.provider_profile,
            "member_profile": obs.member_profile,
        },
    }


# For running directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
