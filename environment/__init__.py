"""
Healthcare Claims Fraud Detection Environment

An OpenEnv-compatible RL environment for sequential fraud detection
in healthcare insurance claims.
"""

# Core types — always importable; no heavy dependencies required.
# The FastAPI server (environment/server/app.py) only needs these.
from .models import (
    ClaimAction,
    ClaimObservation,
    ClaimState,
    InvestigationResult,
    StepResult,
    DecisionType,
    RewardConfig,
    RewardComponents,
)
from .claims_simulator import ClaimsSimulator, SimulatorConfig

# Optional heavy modules (need `requests`, `pandas`, etc.) are NOT imported here.
# Import them explicitly when needed:
#   from environment.client import HealthClaimEnv, LocalHealthClaimEnv
#   from environment.data_loader import UnifiedDataLoader, ...
#   from environment.mixed_data_loader import MixedDataLoader, ...
# This keeps the package importable in lean server/container environments.

__all__ = [
    # Core models
    "ClaimAction",
    "ClaimObservation",
    "ClaimState",
    "InvestigationResult",
    "StepResult",
    "DecisionType",
    "RewardConfig",
    "RewardComponents",
    # Simulator
    "ClaimsSimulator",
    "SimulatorConfig",
]

__version__ = "0.1.0"
