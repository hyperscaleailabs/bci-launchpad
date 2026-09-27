"""Inference: framework-free `Predictor`, Ray batch scoring, Ray Serve deployment.

Only `Predictor` is imported eagerly; the Ray-backed entry points are resolved
lazily so ``import bci_platform.inference`` never requires Ray::

    from bci_platform.inference import predict_pool        # -> inference.batch
    from bci_platform.inference import build_app, run      # -> inference.serve
"""

from importlib import import_module
from typing import Any

from bci_platform.inference.predictor import Predictor, load_predictor

_LAZY = {
    "predict_pool": "bci_platform.inference.batch",
    "predict_pool_local": "bci_platform.inference.batch",
    "PredictorActor": "bci_platform.inference.batch",
    "build_app": "bci_platform.inference.serve",
    "app_builder": "bci_platform.inference.serve",
    "run": "bci_platform.inference.serve",
    "SurrogateModelDeployment": "bci_platform.inference.serve",
}

__all__ = ["Predictor", "load_predictor", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
