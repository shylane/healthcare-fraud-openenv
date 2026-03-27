"""
Comprehensive Data Source Analysis for Healthcare Fraud Detection

This module provides loaders for multiple public healthcare datasets:
1. CMS DE-SynPUF (Medicare claims) - Current
2. Synthea (Synthetic patient records) - NEW
3. HCUP NIS (Hospital data) - Optional
4. MEPS (Medical Expenditure Panel) - Optional

Each loader normalizes data to our ClaimObservation schema.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass


@dataclass
class DataSourceConfig:
    """Configuration for any data source."""

    source_type: str  # 'synpuf', 'synthea', 'hcup', 'meps'
    data_path: str
    sample_size: int = 10000
    seed: int = 42


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self._df: Optional[pd.DataFrame] = None
        self._load_data()

    @abstractmethod
    def _load_data(self):
        """Load and preprocess data from source."""
        pass

    @abstractmethod
    def sample_claim(self, rng) -> Dict[str, Any]:
        """Sample a claim and return normalized dict."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about loaded data."""
        return {
            "source_type": self.config.source_type,
            "total_records": len(self._df) if self._df is not None else 0,
            "sample_size": self.config.sample_size,
        }


class SynPUFDataLoader(BaseDataLoader):
    """
    Loader for CMS DE-SynPUF (Medicare claims data).

    Pros:
    - Real Medicare billing patterns
    - HCPCS/CPT codes (billing codes)
    - Actual dollar amounts
    - Provider NPIs

    Cons:
    - ICD-9 codes (older standard)
    - Limited clinical context
    - Static snapshot (no temporal evolution)
    """

    def _load_data(self):
        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"SynPUF data not found at {path}")

        self._df = pd.read_csv(path, nrows=self.config.sample_size)
        self._preprocess()

    def _preprocess(self):
        """Map SynPUF columns to our schema."""
        # Payment columns
        payment_cols = [c for c in self._df.columns if "PMT_AMT" in c]
        if payment_cols:
            self._df["claim_amount"] = self._df[payment_cols].sum(axis=1)
        else:
            self._df["claim_amount"] = np.random.uniform(50, 500, len(self._df))

        # IDs
        self._df["member_id"] = self._df.get(
            "DESYNPUF_ID", [f"MEM{i}" for i in range(len(self._df))]
        )
        self._df["provider_id"] = self._df.get(
            "PRF_PHYSN_NPI_1", [f"PRV{i}" for i in range(len(self._df))]
        )

        # Codes
        hcpcs_cols = [c for c in self._df.columns if "HCPCS" in c]
        self._df["procedure_codes"] = self._df[hcpcs_cols].fillna("").values.tolist()
        self._df["procedure_codes"] = self._df["procedure_codes"].apply(
            lambda x: [str(c) for c in x if str(c).strip()]
        )

        icd_cols = [c for c in self._df.columns if "DGNS_CD" in c]
        self._df["diagnosis_codes"] = self._df[icd_cols].fillna("").values.tolist()
        self._df["diagnosis_codes"] = self._df["diagnosis_codes"].apply(
            lambda x: [str(c) for c in x if str(c).strip()]
        )

        # Dates
        if "CLM_FROM_DT" in self._df.columns:
            self._df["service_date"] = (
                pd.to_datetime(self._df["CLM_FROM_DT"], format="%Y%m%d", errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("2024-01-01")
            )
        else:
            self._df["service_date"] = "2024-01-01"

        self._df["submission_date"] = self._df["service_date"]
        self._df["place_of_service"] = "11"

        # Calculate profiles
        self._calculate_profiles()

    def _calculate_profiles(self):
        """Calculate provider and member statistics."""
        self.provider_stats = self._df.groupby("provider_id").agg(
            {"claim_amount": ["count", "mean", "sum"]}
        )
        self.member_stats = self._df.groupby("member_id").agg({"claim_amount": ["count", "sum"]})

    def sample_claim(self, rng) -> Dict[str, Any]:
        idx = rng.randint(0, len(self._df))
        row = self._df.iloc[idx]

        provider_id = row["provider_id"]
        member_id = row["member_id"]

        p_stats = self.provider_stats.loc[provider_id]["claim_amount"]
        m_stats = self.member_stats.loc[member_id]["claim_amount"]

        return {
            "claim_id": str(row.get("CLM_ID", f"CLM{idx}")),
            "claim_amount": float(row["claim_amount"]),
            "procedure_codes": row["procedure_codes"],
            "diagnosis_codes": row["diagnosis_codes"],
            "place_of_service": row["place_of_service"],
            "service_date": row["service_date"],
            "submission_date": row["submission_date"],
            "provider_profile": {
                "provider_id": str(provider_id),
                "specialty": "General Practice",
                "total_claims_30d": int(p_stats["count"]),
                "total_amount_30d": float(p_stats["sum"]),
                "avg_claim_amount": float(p_stats["mean"]),
                "claim_denial_rate": 0.05,
                "fraud_flag_rate": 0.01,
                "weekend_claim_rate": 0.1,
                "high_cost_procedure_rate": 0.1,
            },
            "member_profile": {
                "member_id": str(member_id),
                "age": 65,
                "gender": "Unknown",
                "chronic_condition_count": 2,
                "total_claims_90d": int(m_stats["count"]),
                "total_amount_90d": float(m_stats["sum"]),
                "unique_providers_90d": 1,
                "er_visit_count_90d": 0,
                "avg_days_between_claims": 30.0,
            },
        }


class SyntheaDataLoader(BaseDataLoader):
    """
    Loader for Synthea synthetic patient data.

    Pros:
    - Complete patient medical history
    - FHIR R4 format (modern standard)
    - ICD-10 codes (current standard)
    - Temporal patient journeys
    - Medications, allergies, conditions
    - Better for medical necessity fraud detection

    Cons:
    - Synthetic (not real billing patterns)
    - Need to convert encounters to claims
    - No actual dollar amounts (must estimate)

    Best for: Detecting fraud patterns that require clinical context
    """

    def _load_data(self):
        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Synthea data not found at {path}")

        # Synthea exports multiple CSVs or FHIR JSON
        if path.suffix == ".json":
            self._load_fhir_json(path)
        else:
            self._load_synthea_csvs(path)

    def _load_fhir_json(self, path: Path):
        """Load FHIR Bundle JSON."""
        with open(path) as f:
            bundle = json.load(f)

        # Extract encounters (visits) and convert to claims
        encounters = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Encounter":
                encounters.append(self._fhir_encounter_to_claim(resource))

        self._df = pd.DataFrame(encounters)
        self._calculate_synthea_profiles()

    def _load_synthea_csvs(self, path: Path):
        """Load Synthea CSV export folder."""
        # Synthea generates: patients.csv, encounters.csv, conditions.csv, procedures.csv
        encounters_file = path / "encounters.csv"
        if not encounters_file.exists():
            raise FileNotFoundError(f"Synthea encounters.csv not found in {path}")

        encounters = pd.read_csv(encounters_file, nrows=self.config.sample_size)

        # Load related data
        conditions_file = path / "conditions.csv"
        procedures_file = path / "procedures.csv"

        conditions = pd.read_csv(conditions_file) if conditions_file.exists() else None
        procedures = pd.read_csv(procedures_file) if procedures_file.exists() else None

        # Merge and transform
        self._df = self._transform_synthea_encounters(encounters, conditions, procedures)
        self._calculate_synthea_profiles()

    def _fhir_encounter_to_claim(self, encounter: Dict) -> Dict:
        """Convert FHIR Encounter to claim format."""
        # Extract billing codes from encounter
        type_coding = encounter.get("type", [{}])[0].get("coding", [{}])[0]
        procedure_code = type_coding.get("code", "99213")

        # Estimate cost based on encounter type
        cost_map = {
            "ambulatory": 150.0,
            "emergency": 800.0,
            "inpatient": 2500.0,
            "outpatient": 300.0,
        }
        encounter_class = encounter.get("class", {}).get("code", "ambulatory")
        base_cost = cost_map.get(encounter_class, 200.0)

        return {
            "claim_id": encounter.get("id", "UNKNOWN"),
            "member_id": encounter.get("subject", {})
            .get("reference", "PATIENT")
            .replace("urn:uuid:", ""),
            "provider_id": encounter.get("serviceProvider", {})
            .get("reference", "PROVIDER")
            .replace("urn:uuid:", ""),
            "procedure_codes": [procedure_code],
            "diagnosis_codes": [],  # Would need to look up conditions
            "claim_amount": base_cost * np.random.uniform(0.8, 1.2),
            "service_date": encounter.get("period", {}).get("start", "2024-01-01")[:10],
            "place_of_service": self._map_encounter_class_to_pos(encounter_class),
        }

    def _transform_synthea_encounters(
        self,
        encounters: pd.DataFrame,
        conditions: Optional[pd.DataFrame],
        procedures: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Transform Synthea encounters to claim format."""
        claims = []

        for _, enc in encounters.iterrows():
            claim = {
                "claim_id": enc.get("Id", f"CLM{len(claims)}"),
                "member_id": enc.get("PATIENT", "UNKNOWN"),
                "provider_id": enc.get("PROVIDER", "UNKNOWN"),
                "service_date": enc.get("START", "2024-01-01")[:10],
                "procedure_codes": [],
                "diagnosis_codes": [],
                "claim_amount": enc.get("BASE_ENCOUNTER_COST", 200.0) * np.random.uniform(0.9, 1.1),
                "place_of_service": self._map_encounter_class_to_pos(
                    enc.get("ENCOUNTERCLASS", "ambulatory")
                ),
            }

            # Add procedures if available
            if procedures is not None:
                enc_procs = procedures[procedures["ENCOUNTER"] == enc["Id"]]
                claim["procedure_codes"] = enc_procs["CODE"].astype(str).tolist()[:3]

            # Add conditions if available
            if conditions is not None:
                # Find conditions for this patient around this time
                patient_conds = conditions[conditions["PATIENT"] == enc["PATIENT"]]
                claim["diagnosis_codes"] = patient_conds["CODE"].astype(str).tolist()[:2]

            claims.append(claim)

        return pd.DataFrame(claims)

    def _map_encounter_class_to_pos(self, encounter_class: str) -> str:
        """Map Synthea encounter class to Place of Service code."""
        mapping = {
            "ambulatory": "11",  # Office
            "emergency": "23",  # ER
            "inpatient": "21",  # Inpatient hospital
            "outpatient": "22",  # Outpatient hospital
            "wellness": "11",  # Office
            "urgentcare": "20",  # Urgent care
            "home": "12",  # Home
        }
        return mapping.get(encounter_class.lower(), "11")

    def _calculate_synthea_profiles(self):
        """Calculate statistics from Synthea data."""
        self.provider_stats = self._df.groupby("provider_id").agg(
            {"claim_amount": ["count", "mean", "sum"]}
        )
        self.member_stats = self._df.groupby("member_id").agg({"claim_amount": ["count", "sum"]})

    def sample_claim(self, rng) -> Dict[str, Any]:
        idx = rng.randint(0, len(self._df))
        row = self._df.iloc[idx]

        provider_id = row["provider_id"]
        member_id = row["member_id"]

        # Get stats (with fallback for new providers/members)
        if provider_id in self.provider_stats.index:
            p_stats = self.provider_stats.loc[provider_id]["claim_amount"]
        else:
            p_stats = {"count": 0, "mean": 200.0, "sum": 0.0}

        if member_id in self.member_stats.index:
            m_stats = self.member_stats.loc[member_id]["claim_amount"]
        else:
            m_stats = {"count": 0, "sum": 0.0}

        return {
            "claim_id": str(row["claim_id"]),
            "claim_amount": float(row["claim_amount"]),
            "procedure_codes": row["procedure_codes"] if row["procedure_codes"] else ["99213"],
            "diagnosis_codes": row["diagnosis_codes"] if row["diagnosis_codes"] else ["Z00.00"],
            "place_of_service": row["place_of_service"],
            "service_date": row["service_date"],
            "submission_date": row["service_date"],
            "provider_profile": {
                "provider_id": str(provider_id),
                "specialty": "General Practice",  # Could look up from providers file
                "total_claims_30d": int(p_stats["count"]),
                "total_amount_30d": float(p_stats["sum"]),
                "avg_claim_amount": float(p_stats["mean"]) if p_stats["mean"] > 0 else 200.0,
                "claim_denial_rate": 0.05,
                "fraud_flag_rate": 0.01,
                "weekend_claim_rate": 0.1,
                "high_cost_procedure_rate": 0.1,
            },
            "member_profile": {
                "member_id": str(member_id),
                "age": 45,  # Could look up from patients file
                "gender": "Unknown",
                "chronic_condition_count": len(row.get("diagnosis_codes", [])),
                "total_claims_90d": int(m_stats["count"]),
                "total_amount_90d": float(m_stats["sum"]),
                "unique_providers_90d": 1,
                "er_visit_count_90d": 1 if row["place_of_service"] == "23" else 0,
                "avg_days_between_claims": 30.0
                if m_stats["count"] == 0
                else 90.0 / max(m_stats["count"], 1),
            },
        }


class UnifiedDataLoader:
    """
    Unified loader that can work with any data source.

    Usage:
        # SynPUF mode
        config = DataSourceConfig(source_type='synpuf', data_path='data/synpuf.csv')
        loader = UnifiedDataLoader(config)

        # Synthea mode
        config = DataSourceConfig(source_type='synthea', data_path='data/synthea/')
        loader = UnifiedDataLoader(config)
    """

    def __init__(self, config: DataSourceConfig):
        self.config = config

        if config.source_type == "synpuf":
            self._loader = SynPUFDataLoader(config)
        elif config.source_type == "synthea":
            self._loader = SyntheaDataLoader(config)
        else:
            raise ValueError(f"Unknown source type: {config.source_type}")

    def sample_claim(self, rng) -> Dict[str, Any]:
        return self._loader.sample_claim(rng)

    def get_statistics(self) -> Dict[str, Any]:
        stats = self._loader.get_statistics()
        stats["loader_type"] = self.config.source_type
        return stats


# Backwards compatibility
RealDataLoader = SynPUFDataLoader
DataLoaderConfig = DataSourceConfig
