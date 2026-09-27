"""Experiment tracking + model registry adapter (the only package that imports mlflow)."""

from bci_platform.tracking.mlflow_client import (
    Tracker,
    environment_metadata,
    resolve_tracking_uri,
)
from bci_platform.tracking.registry import (
    ModelRegistry,
    NoProductionModelError,
    ProductionModel,
    PromotionError,
)

__all__ = [
    "ModelRegistry",
    "NoProductionModelError",
    "ProductionModel",
    "PromotionError",
    "Tracker",
    "environment_metadata",
    "resolve_tracking_uri",
]
