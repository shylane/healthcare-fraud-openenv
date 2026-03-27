"""
Claims Simulator for the Healthcare Claims Fraud Detection Environment.

Generates realistic synthetic healthcare claims with configurable fraud patterns
based on patterns observed in CMS data and healthcare fraud literature.
"""

import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .data_loader import RealDataLoader

from .models import (
    ClaimObservation,
    ProviderProfile,
    MemberProfile,
    RewardConfig,
)


# Common procedure codes (CPT/HCPCS) by category
PROCEDURE_CODES = {
    "office_visit": ["99213", "99214", "99215", "99203", "99204", "99205"],
    "hospital_inpatient": ["99221", "99222", "99223", "99231", "99232", "99233"],
    "emergency": ["99281", "99282", "99283", "99284", "99285"],
    "lab": ["80053", "85025", "80061", "84443", "82465"],
    "imaging": ["71046", "72148", "73721", "74176", "70553"],
    "surgery_minor": ["10060", "11102", "17000", "20610", "29881"],
    "surgery_major": ["27447", "33533", "44140", "47562", "63030"],
    "physical_therapy": ["97110", "97140", "97530", "97542", "97750"],
    "dme": ["E0601", "E0260", "E0781", "K0001", "A4253"],
    "mental_health": ["90834", "90837", "90847", "90853", "96132"],
}

# Common diagnosis codes (ICD-10)
DIAGNOSIS_CODES = {
    "diabetes": ["E11.9", "E11.65", "E11.40", "E11.21", "E11.22"],
    "hypertension": ["I10", "I11.9", "I12.9", "I13.10", "I15.0"],
    "heart_disease": ["I25.10", "I50.9", "I48.91", "I21.9", "I35.0"],
    "respiratory": ["J44.9", "J45.909", "J18.9", "J06.9", "J20.9"],
    "musculoskeletal": ["M54.5", "M79.3", "M25.50", "M17.11", "M47.812"],
    "mental": ["F32.9", "F41.9", "F43.10", "F10.20", "F31.9"],
    "cancer": ["C34.90", "C50.919", "C61", "C18.9", "C64.9"],
    "injury": ["S72.001A", "S42.001A", "S82.001A", "S52.501A", "S32.000A"],
}

# Places of service
PLACES_OF_SERVICE = {
    "11": "Office",
    "21": "Inpatient Hospital",
    "22": "Outpatient Hospital",
    "23": "Emergency Room",
    "31": "Skilled Nursing Facility",
    "32": "Nursing Facility",
    "51": "Inpatient Psychiatric",
    "65": "End-Stage Renal Dialysis",
    "81": "Independent Laboratory",
}

# Provider specialties
SPECIALTIES = [
    "Internal Medicine",
    "Family Practice",
    "Cardiology",
    "Orthopedic Surgery",
    "General Surgery",
    "Emergency Medicine",
    "Radiology",
    "Anesthesiology",
    "Psychiatry",
    "Physical Therapy",
    "Dermatology",
    "Neurology",
    "Oncology",
    "Urology",
    "Ophthalmology",
]


@dataclass
class FraudPattern:
    """
    Defines a fraud pattern for generating fraudulent claims.

    Each pattern has a name, probability of occurrence, and
    characteristics that make it detectable.
    """

    name: str
    description: str
    detection_difficulty: float  # 0-1, higher = harder to detect

    # Pattern characteristics
    claim_amount_multiplier: float = 1.0
    procedure_code_anomaly: bool = False
    provider_pattern_anomaly: bool = False
    member_pattern_anomaly: bool = False


# Define known fraud patterns
FRAUD_PATTERNS = [
    FraudPattern(
        name="Upcoding",
        description="Billing for a more expensive service than performed",
        detection_difficulty=0.4,
        claim_amount_multiplier=2.5,
        procedure_code_anomaly=True,
    ),
    FraudPattern(
        name="Unbundling",
        description="Billing separately for procedures usually bundled",
        detection_difficulty=0.5,
        claim_amount_multiplier=1.5,
        procedure_code_anomaly=True,
    ),
    FraudPattern(
        name="Phantom Billing",
        description="Billing for services not performed",
        detection_difficulty=0.7,
        claim_amount_multiplier=1.0,
        provider_pattern_anomaly=True,
    ),
    FraudPattern(
        name="Duplicate Billing",
        description="Billing twice for the same service",
        detection_difficulty=0.2,
        claim_amount_multiplier=1.0,
        provider_pattern_anomaly=True,
    ),
    FraudPattern(
        name="Kickbacks",
        description="Referrals in exchange for financial incentives",
        detection_difficulty=0.9,
        claim_amount_multiplier=1.2,
        provider_pattern_anomaly=True,
    ),
    FraudPattern(
        name="Identity Theft",
        description="Using member ID for another person's care",
        detection_difficulty=0.8,
        claim_amount_multiplier=1.0,
        member_pattern_anomaly=True,
    ),
    FraudPattern(
        name="Services Not Medically Necessary",
        description="Billing for unnecessary tests or procedures",
        detection_difficulty=0.6,
        claim_amount_multiplier=1.3,
        procedure_code_anomaly=True,
    ),
    FraudPattern(
        name="Time Padding",
        description="Billing for more time than spent with patient",
        detection_difficulty=0.5,
        claim_amount_multiplier=1.4,
        provider_pattern_anomaly=True,
    ),
]


@dataclass
class SimulatorConfig:
    """Configuration for the claims simulator."""

    # Fraud rate (proportion of claims that are fraudulent)
    fraud_rate: float = 0.05

    # Claim amount distribution parameters
    mean_claim_amount: float = 500.0
    std_claim_amount: float = 800.0
    min_claim_amount: float = 25.0
    max_claim_amount: float = 50000.0

    # Episode configuration
    claims_per_episode: int = 100

    # Provider pool size
    num_providers: int = 200
    num_fraudulent_providers: int = 10

    # Member pool size
    num_members: int = 1000

    # Random seed for reproducibility
    seed: Optional[int] = None

    # Feature noise level (0-1)
    feature_noise: float = 0.1

    # Repeat-provider bias: probability of reusing a recently-seen provider
    # when generating a non-fraud claim.  Higher values make episodes more
    # realistic (same providers submit multiple claims) and give the agent
    # more opportunities to leverage investigation memory.
    repeat_provider_bias: float = 0.30


class ClaimsSimulator:
    """
    Generates realistic synthetic healthcare claims with fraud patterns.

    The simulator maintains state about providers and members to generate
    consistent patterns over time, making fraud detection more realistic.

    Supports "Hybrid Mode" where base claims are sampled from real data
    (via data_loader) and then potentially modified to be fraudulent.
    """

    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        data_loader: Optional["RealDataLoader"] = None,
    ):
        """
        Initialize the claims simulator.

        Args:
            config: Configuration for the simulator. Uses defaults if None.
            data_loader: Optional loader for real data (hybrid mode).
        """
        self.config = config or SimulatorConfig()
        self.data_loader = data_loader

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # Initialize provider and member pools (for synthetic generation)
        self._providers: Dict[str, Dict[str, Any]] = {}
        self._members: Dict[str, Dict[str, Any]] = {}
        self._fraudulent_provider_ids: set = set()

        # Claim history for pattern detection
        self._provider_claim_history: Dict[str, List[Dict]] = {}
        self._member_claim_history: Dict[str, List[Dict]] = {}

        # Track recently-used provider IDs this episode for repeat-provider bias
        self._recent_providers: List[str] = []

        # Initialize pools
        self._initialize_providers()
        self._initialize_members()

    def _generate_id(self, prefix: str = "") -> str:
        """Generate a unique identifier using seeded random."""
        # Use seeded random instead of uuid for reproducibility
        chars = "0123456789ABCDEF"
        random_part = "".join(random.choice(chars) for _ in range(12))
        return f"{prefix}{random_part}"

    def _initialize_providers(self):
        """Initialize the provider pool."""
        # Create regular providers
        for i in range(self.config.num_providers - self.config.num_fraudulent_providers):
            provider_id = self._generate_id("PRV")
            self._providers[provider_id] = {
                "id": provider_id,
                "specialty": random.choice(SPECIALTIES),
                "is_fraudulent": False,
                "billing_pattern": "normal",
                "avg_claim_amount": np.random.lognormal(5.5, 0.8),
                "claims_per_day": np.random.poisson(5),
            }
            self._provider_claim_history[provider_id] = []

        # Create fraudulent providers
        for i in range(self.config.num_fraudulent_providers):
            provider_id = self._generate_id("PRV")
            pattern = random.choice(FRAUD_PATTERNS)
            self._providers[provider_id] = {
                "id": provider_id,
                "specialty": random.choice(SPECIALTIES),
                "is_fraudulent": True,
                "fraud_pattern": pattern,
                "billing_pattern": "fraudulent",
                "avg_claim_amount": np.random.lognormal(6.0, 1.0),
                "claims_per_day": np.random.poisson(8),
            }
            self._fraudulent_provider_ids.add(provider_id)
            self._provider_claim_history[provider_id] = []

    def _initialize_members(self):
        """Initialize the member pool."""
        for i in range(self.config.num_members):
            member_id = self._generate_id("MBR")

            # Generate chronic conditions
            conditions = []
            if random.random() < 0.4:
                conditions.append(random.choice(list(DIAGNOSIS_CODES.keys())))
            if random.random() < 0.15:
                conditions.append(random.choice(list(DIAGNOSIS_CODES.keys())))

            self._members[member_id] = {
                "id": member_id,
                "age": random.randint(18, 90),
                "gender": random.choice(["M", "F"]),
                "chronic_conditions": conditions,
                "risk_score": random.random(),
            }
            self._member_claim_history[member_id] = []

    def _get_provider_profile(self, provider_id: str) -> Dict[str, Any]:
        """Generate provider profile stats."""
        # Check if we have history, otherwise use synthetic defaults
        history = self._provider_claim_history.get(provider_id, [])
        provider = self._providers.get(provider_id, {})

        # Calculate stats from history
        if history:
            amounts = [c["amount"] for c in history]
            total_claims = len(history)
            total_amount = sum(amounts)
            avg_amount = total_amount / total_claims
        else:
            # Cold start stats
            total_claims = 0
            total_amount = 0.0
            avg_amount = provider.get("avg_claim_amount", 0.0)

        # Derived metrics with noise
        is_fraud = provider.get("is_fraudulent", False)

        # Fraudulent providers have higher flag/denial rates
        base_denial = 0.15 if is_fraud else 0.05
        base_flag = 0.10 if is_fraud else 0.01

        return {
            "provider_id": provider_id,
            "specialty": provider.get("specialty", "Unknown"),
            "total_claims_30d": total_claims,
            "total_amount_30d": total_amount,
            "avg_claim_amount": avg_amount,
            "claim_denial_rate": np.clip(base_denial + np.random.normal(0, 0.02), 0, 1),
            "fraud_flag_rate": np.clip(base_flag + np.random.normal(0, 0.01), 0, 1),
            "unique_patients_30d": int(total_claims * 0.8),
            "unique_procedures_30d": int(total_claims * 0.3) + 1,
            "weekend_claim_rate": 0.2 if is_fraud else 0.05,
            "high_cost_procedure_rate": 0.3 if is_fraud else 0.1,
        }

    def _get_member_profile(self, member_id: str) -> Dict[str, Any]:
        """Generate member profile stats."""
        history = self._member_claim_history.get(member_id, [])
        member = self._members.get(member_id, {})

        if history:
            amounts = [c["amount"] for c in history]
            total_claims = len(history)
            total_amount = sum(amounts)
        else:
            total_claims = 0
            total_amount = 0.0

        return {
            "member_id": member_id,
            "age": member.get("age", 45),
            "gender": member.get("gender", "U"),
            "chronic_condition_count": len(member.get("chronic_conditions", [])),
            "total_claims_90d": total_claims,
            "total_amount_90d": total_amount,
            "unique_providers_90d": min(total_claims, 5),
            "er_visit_count_90d": int(total_claims * 0.1),
            "prescription_count_90d": int(total_claims * 2),
            "avg_days_between_claims": 30.0 if total_claims == 0 else 90.0 / total_claims,
        }

    def _calculate_features(
        self,
        claim_amount: float,
        provider_profile: Dict[str, Any],
        member_profile: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate derived fraud features."""
        # Z-score of claim amount vs provider average
        avg_amt = provider_profile["avg_claim_amount"]
        if avg_amt > 0:
            amount_zscore = (claim_amount - avg_amt) / (avg_amt * 0.5 + 1.0)
        else:
            amount_zscore = 0.0

        # Provider risk score
        provider_risk_score = (
            provider_profile["fraud_flag_rate"] * 0.5
            + provider_profile["claim_denial_rate"] * 0.3
            + (1.0 if provider_profile["weekend_claim_rate"] > 0.2 else 0.0) * 0.2
        )

        features = {
            "amount_zscore": float(amount_zscore),
            "provider_risk_score": float(provider_risk_score),
            "member_risk_score": float(member_profile["total_claims_90d"] * 0.05),
            "dx_proc_consistency": random.uniform(0.5, 1.0),  # Simplified
            "time_since_last_claim": member_profile["avg_days_between_claims"],
        }

        # Add noise
        noise = self.config.feature_noise
        for key in features:
            features[key] += np.random.normal(0, noise)

        return features

    def generate_claim(
        self,
        force_fraud: Optional[bool] = None,
        claim_date: Optional[datetime] = None,
    ) -> Tuple[ClaimObservation, bool]:
        """
        Generate a single claim (either synthetic or sampled real).

        Args:
            force_fraud: If True/False, force fraud status. If None, use fraud_rate.
            claim_date: Date for the claim. If None, use current date.

        Returns:
            Tuple of (ClaimObservation, is_fraud)
        """
        # Determine if this claim is fraudulent
        if force_fraud is not None:
            is_fraud = force_fraud
        else:
            is_fraud = random.random() < self.config.fraud_rate

        # HYBRID MODE: If data loader is present, sample from real data
        if self.data_loader:
            claim_dict = self.data_loader.sample_claim(np.random)

            # If fraud, apply injection logic
            if is_fraud:
                # 1. Multiplier on amount
                multiplier = random.uniform(1.5, 4.0)
                claim_dict["claim_amount"] *= multiplier

                # 2. Add high-risk code if not present
                if random.random() < 0.5:
                    claim_dict["procedure_codes"].append("99215")  # High level office visit

                # 3. Update profiles to look suspicious
                claim_dict["provider_profile"]["fraud_flag_rate"] = 0.15
                claim_dict["provider_profile"]["avg_claim_amount"] = (
                    claim_dict["claim_amount"] / multiplier
                )

                # 4. Calculate features based on modified data
                claim_dict["claim_features"] = self._calculate_features(
                    claim_dict["claim_amount"],
                    claim_dict["provider_profile"],
                    claim_dict["member_profile"],
                )
            else:
                # Legitimate claim - clean up profiles
                claim_dict["provider_profile"]["fraud_flag_rate"] = 0.01
                claim_dict["claim_features"] = self._calculate_features(
                    claim_dict["claim_amount"],
                    claim_dict["provider_profile"],
                    claim_dict["member_profile"],
                )

            return ClaimObservation(**claim_dict), is_fraud

        # SYNTHETIC MODE (Fallback if no data loader)

        # Select provider — with repeat-provider bias for realism.
        # 1. Fraud claims: 70% chance of known-fraudulent provider (existing logic).
        # 2. Non-fraud claims: ``repeat_provider_bias`` chance of reusing a
        #    recently-seen provider so the agent encounters the same provider
        #    multiple times per episode (enabling memory-based decisions).
        if is_fraud and random.random() < 0.7:
            # 70% of fraud comes from known fraudulent providers
            provider_id = random.choice(list(self._fraudulent_provider_ids))
        elif self._recent_providers and random.random() < self.config.repeat_provider_bias:
            # Repeat-provider bias: reuse a recent provider
            provider_id = random.choice(self._recent_providers)
        else:
            provider_id = random.choice(list(self._providers.keys()))

        provider = self._providers[provider_id]

        # Select member
        member_id = random.choice(list(self._members.keys()))
        member = self._members[member_id]

        # Generate claim date
        if claim_date is None:
            claim_date = datetime.now() - timedelta(days=random.randint(0, 30))

        service_date = claim_date - timedelta(days=random.randint(0, 14))
        is_weekend = service_date.weekday() >= 5

        # Select place of service
        if provider["specialty"] == "Emergency Medicine":
            pos = "23"
        elif provider["specialty"] in ["Orthopedic Surgery", "General Surgery"]:
            pos = random.choice(["21", "22", "11"])
        else:
            pos = random.choice(["11", "22", "81"])

        # Generate procedures based on specialty
        specialty_procs = {
            "Cardiology": ["office_visit", "lab", "imaging"],
            "Orthopedic Surgery": ["surgery_minor", "surgery_major", "imaging"],
            "Emergency Medicine": ["emergency", "lab", "imaging"],
            "Physical Therapy": ["physical_therapy"],
            "Psychiatry": ["mental_health", "office_visit"],
            "Radiology": ["imaging"],
        }

        proc_categories = specialty_procs.get(provider["specialty"], ["office_visit", "lab"])

        procedures = []
        for cat in random.sample(
            proc_categories, k=min(len(proc_categories), random.randint(1, 3))
        ):
            if cat in PROCEDURE_CODES:
                procedures.append(random.choice(PROCEDURE_CODES[cat]))

        # Generate diagnoses based on member conditions
        diagnoses = []
        for condition in member.get("chronic_conditions", [])[:2]:
            if condition in DIAGNOSIS_CODES:
                diagnoses.append(random.choice(DIAGNOSIS_CODES[condition]))
        if not diagnoses:
            diagnoses.append(random.choice(DIAGNOSIS_CODES["musculoskeletal"]))

        # Generate claim amount
        base_amount = np.random.lognormal(np.log(self.config.mean_claim_amount), 0.8)

        # Apply fraud multiplier if fraudulent
        if is_fraud and provider.get("fraud_pattern"):
            pattern = provider["fraud_pattern"]
            base_amount *= pattern.claim_amount_multiplier

        claim_amount = float(
            np.clip(base_amount, self.config.min_claim_amount, self.config.max_claim_amount)
        )

        # Get profiles
        provider_profile = self._get_provider_profile(provider_id)
        member_profile = self._get_member_profile(member_id)

        # Generate derived features
        features = self._calculate_features(claim_amount, provider_profile, member_profile)

        # Update histories (for next time)
        claim_record = {"amount": claim_amount, "date": service_date, "is_fraud": is_fraud}
        self._provider_claim_history[provider_id].append(claim_record)
        self._member_claim_history[member_id].append(claim_record)

        # Track this provider for repeat-provider bias
        self._recent_providers.append(provider_id)

        return ClaimObservation(
            claim_id=self._generate_id("CLM"),
            claim_amount=round(claim_amount, 2),
            procedure_codes=procedures,
            diagnosis_codes=diagnoses,
            place_of_service=pos,
            service_date=service_date.strftime("%Y-%m-%d"),
            submission_date=claim_date.strftime("%Y-%m-%d"),
            provider_profile=provider_profile,
            member_profile=member_profile,
            claim_features=features,
        ), is_fraud

    def generate_batch(self, size: int) -> List[Tuple[ClaimObservation, bool]]:
        """Generate a batch of claims."""
        return [self.generate_claim() for _ in range(size)]

    def reset(self):
        """Reset claim histories and re-seed RNG for reproducibility."""
        # Re-seed for reproducibility between episodes
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        for provider_id in self._provider_claim_history:
            self._provider_claim_history[provider_id] = []
        for member_id in self._member_claim_history:
            self._member_claim_history[member_id] = []

        # Clear repeat-provider tracking for new episode
        self._recent_providers = []

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulator statistics."""
        total_provider_claims = sum(len(h) for h in self._provider_claim_history.values())
        fraud_provider_claims = sum(
            len(self._provider_claim_history.get(pid, [])) for pid in self._fraudulent_provider_ids
        )

        return {
            "mode": "hybrid" if self.data_loader else "synthetic",
            "num_providers": len(self._providers),
            "num_fraudulent_providers": len(self._fraudulent_provider_ids),
            "num_members": len(self._members),
            "total_claims_generated": total_provider_claims,
            "fraud_provider_claims": fraud_provider_claims,
            "fraud_rate": self.config.fraud_rate,
        }
