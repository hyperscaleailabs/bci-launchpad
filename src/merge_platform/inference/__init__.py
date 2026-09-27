"""Inference: framework-free `Predictor` (batch/serve modules live alongside, owned elsewhere)."""

from merge_platform.inference.predictor import Predictor, load_predictor

__all__ = ["Predictor", "load_predictor"]
