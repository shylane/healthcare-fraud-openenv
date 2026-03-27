"""
Mixed Data Loader - Combines Multiple Data Sources

This loader samples from both Synthea and SynPUF (or any other sources)
to create a hybrid training dataset with the best of both worlds.

Key Design Decisions:
1. NO ETL - Runtime mixing is more flexible and efficient
2. NO schema changes - ClaimObservation remains clean and consistent
3. Source tracking via logging/metadata - Separation of concerns
4. Dynamic ratios - Can adjust mix without reprocessing

Usage:
    config = MixedDataConfig(
        sources=[
            {'type': 'synthea', 'path': 'data/synthea/', 'weight': 0.6},
            {'type': 'synpuf', 'path': 'data/synpuf.csv', 'weight': 0.4}
        ]
    )
    loader = MixedDataLoader(config)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .data_loader import (
    UnifiedDataLoader,
    DataSourceConfig,
    SynPUFDataLoader,
    SyntheaDataLoader,
)

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class SourceTrackingMetrics:
    """Tracks sampling statistics per source."""

    source_name: str
    samples_drawn: int = 0
    fraud_injected: int = 0
    total_claim_amount: float = 0.0


@dataclass
class MixedDataConfig:
    """Configuration for mixed data loading."""

    sources: List[
        Dict[str, Any]
    ]  # List of {'type': 'synthea|synpuf', 'path': '...', 'weight': 0.5}
    seed: int = 42
    track_sources: bool = True  # Enable detailed source tracking


class MixedDataLoader:
    """
    Loads data from multiple sources with configurable mixing ratios.

    Design Philosophy:
    - Runtime mixing > ETL (more flexible, no preprocessing)
    - Schema consistency > Source attribution in data (cleaner design)
    - Logging > Schema bloat (track sources without polluting data model)

    Example:
        config = MixedDataConfig(sources=[
            {'type': 'synthea', 'path': 'data/synthea/', 'weight': 0.6},
            {'type': 'synpuf', 'path': 'data/synpuf.csv', 'weight': 0.4}
        ])
        loader = MixedDataLoader(config)

        # 60% of samples will come from Synthea, 40% from SynPUF
        claim = loader.sample_claim(rng)
    """

    def __init__(self, config: MixedDataConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Initialize loaders for each source
        self.loaders = []
        self.weights = []
        self.source_names = []

        for source_config in config.sources:
            source_type = source_config["type"]
            path = source_config["path"]
            weight = source_config["weight"]

            # Create individual loader
            loader_config = DataSourceConfig(
                source_type=source_type, data_path=path, seed=config.seed
            )
            loader = UnifiedDataLoader(loader_config)

            self.loaders.append(loader)
            self.weights.append(weight)
            self.source_names.append(source_type)

        # Normalize weights to sum to 1
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        # Initialize tracking metrics if enabled
        if config.track_sources:
            self.metrics = {
                name: SourceTrackingMetrics(source_name=name) for name in self.source_names
            }
            self.sampling_history: List[Tuple[str, str]] = []  # (claim_id, source)
        else:
            self.metrics = None
            self.sampling_history = None

        logger.info(f"MixedDataLoader initialized with {len(self.loaders)} sources:")
        for i, (loader, weight, name) in enumerate(
            zip(self.loaders, self.weights, self.source_names)
        ):
            stats = loader.get_statistics()
            logger.info(
                f"  Source {i + 1}: {name} (weight: {weight:.1%}, records: {stats['total_records']})"
            )

    def sample_claim(self, rng) -> Dict[str, Any]:
        """
        Sample a claim from one of the sources based on weights.

        Args:
            rng: Random number generator

        Returns:
            Claim dictionary (matches ClaimObservation schema)
        """
        # Select source based on weights
        source_idx = self.rng.choice(len(self.loaders), p=self.weights)
        selected_loader = self.loaders[source_idx]
        source_name = self.source_names[source_idx]

        # Sample from selected loader
        claim = selected_loader.sample_claim(rng)

        # Track metrics if enabled
        if self.config.track_sources and self.metrics:
            self.metrics[source_name].samples_drawn += 1
            self.metrics[source_name].total_claim_amount += claim.get("claim_amount", 0)
            self.sampling_history.append((claim.get("claim_id", "unknown"), source_name))

            # Log every 100 samples for monitoring
            if len(self.sampling_history) % 100 == 0:
                self._log_sampling_stats()

        return claim

    def _log_sampling_stats(self):
        """Log current sampling statistics."""
        if not self.metrics:
            return

        total_samples = sum(m.samples_drawn for m in self.metrics.values())
        if total_samples == 0:
            return

        logger.info(f"Sampling statistics after {total_samples} claims:")
        for name, metrics in self.metrics.items():
            actual_ratio = metrics.samples_drawn / total_samples
            target_ratio = self.weights[self.source_names.index(name)]
            logger.info(
                f"  {name}: {metrics.samples_drawn} samples "
                f"(actual: {actual_ratio:.1%}, target: {target_ratio:.1%})"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from all sources."""
        stats = {
            "mode": "mixed",
            "num_sources": len(self.loaders),
            "target_weights": dict(zip(self.source_names, self.weights)),
            "sources": [],
        }

        # Add actual sampling ratios if tracking enabled
        if self.metrics:
            total_samples = sum(m.samples_drawn for m in self.metrics.values())
            if total_samples > 0:
                stats["actual_ratios"] = {
                    name: m.samples_drawn / total_samples for name, m in self.metrics.items()
                }

        for loader, weight in zip(self.loaders, self.weights):
            source_stats = loader.get_statistics()
            source_stats["sampling_weight"] = weight
            stats["sources"].append(source_stats)

        return stats

    def get_source_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed breakdown of sampling by source.

        Returns:
            Dictionary with per-source metrics
        """
        if not self.metrics:
            return {"error": "Source tracking not enabled"}

        return {
            name: {
                "samples_drawn": m.samples_drawn,
                "total_claim_amount": m.total_claim_amount,
                "avg_claim_amount": (
                    m.total_claim_amount / m.samples_drawn if m.samples_drawn > 0 else 0
                ),
            }
            for name, m in self.metrics.items()
        }

    def reset_tracking(self):
        """Reset tracking metrics (useful between episodes)."""
        if self.metrics:
            for m in self.metrics.values():
                m.samples_drawn = 0
                m.total_claim_amount = 0.0
        if self.sampling_history:
            self.sampling_history.clear()


# Convenience function for the recommended 60/40 mix
def create_recommended_mixed_loader(
    synthea_path: str = "data/synthea/",
    synpuf_path: str = "data/raw/carrier_claims_sample.csv",
    synthea_weight: float = 0.6,
    synpuf_weight: float = 0.4,
    seed: int = 42,
    track_sources: bool = True,
) -> MixedDataLoader:
    """
    Create the recommended mixed loader: 60% Synthea, 40% SynPUF

    This mix provides:
    - 60% Synthea: Rich clinical context for LLM reasoning
    - 40% SynPUF: Real Medicare billing patterns

    Rationale for this ratio:
    - Synthea provides the clinical context needed for sophisticated fraud detection
    - SynPUF grounds the model in real-world billing patterns
    - 60/40 balances innovation (clinical reasoning) with realism (billing patterns)

    Args:
        synthea_path: Path to Synthea data folder
        synpuf_path: Path to SynPUF CSV file
        synthea_weight: Weight for Synthea (default 0.6)
        synpuf_weight: Weight for SynPUF (default 0.4)
        seed: Random seed
        track_sources: Enable detailed source tracking

    Returns:
        MixedDataLoader configured with recommended weights
    """
    config = MixedDataConfig(
        sources=[
            {"type": "synthea", "path": synthea_path, "weight": synthea_weight},
            {"type": "synpuf", "path": synpuf_path, "weight": synpuf_weight},
        ],
        seed=seed,
        track_sources=track_sources,
    )
    return MixedDataLoader(config)
