"""
Healthcare Claims Fraud Detection Environment

An OpenEnv-compatible RL environment for sequential fraud detection
in healthcare insurance claims.
"""

from .models import (
    ClaimAction,
    ClaimObservation,
    ClaimState,
    InvestigationResult,
    StepResult,
    DecisionType,  # Renamed from ActionType
    RewardConfig,
    RewardComponents,
)
from .client import HealthClaimEnv, LocalHealthClaimEnv
from .claims_simulator import ClaimsSimulator, SimulatorConfig
from .data_loader import (
    UnifiedDataLoader,
    DataSourceConfig,
    SynPUFDataLoader,
    SyntheaDataLoader,
)
from .mixed_data_loader import (
    MixedDataLoader,
    MixedDataConfig,
    create_recommended_mixed_loader,
)

__all__ = [
    # Models
    "ClaimAction",
    "ClaimObservation",
    "ClaimState",
    "InvestigationResult",
    "StepResult",
    "DecisionType",
    "RewardConfig",
    "RewardComponents",
    # Clients
    "HealthClaimEnv",
    "LocalHealthClaimEnv",
    # Simulator
    "ClaimsSimulator",
    "SimulatorConfig",
    # Data Loaders
    "UnifiedDataLoader",
    "DataSourceConfig",
    "SynPUFDataLoader",
    "SyntheaDataLoader",
    "MixedDataLoader",
    "MixedDataConfig",
    "create_recommended_mixed_loader",
]

__version__ = "0.1.0"
