"""Typed platform configuration.

`PlatformConfig` is the single source of truth for every tunable in the demo.
It is loaded from `configs/{local,distributed,gpu}.yaml`; the environment
variable ``BCI_CONFIG`` selects the file when no explicit path is given.

Layering: a config file may declare ``extends: local.yaml`` to inherit from
another file (resolved relative to itself) and override only what differs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "local.yaml"
CONFIG_ENV_VAR = "BCI_CONFIG"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class PathsConfig(_Section):
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    artifacts_dir: Path = Path("artifacts")

    def resolved(self, base: Path | None = None) -> PathsConfig:
        """Return a copy with relative paths anchored at ``base`` (default: repo root)."""
        base = base or REPO_ROOT
        return PathsConfig(
            data_dir=_anchor(self.data_dir, base),
            reports_dir=_anchor(self.reports_dir, base),
            artifacts_dir=_anchor(self.artifacts_dir, base),
        )


def _anchor(p: Path, base: Path) -> Path:
    return p if p.is_absolute() else (base / p)


class DataConfig(_Section):
    n_features: int = 32
    pool_size: int = 100_000
    initial_observations: int = 500
    batch_size_per_round: int = 100
    pool_seed: int = 0


class ModelConfig(_Section):
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256, 128])
    dropout: float = 0.1


class TrainingConfig(_Section):
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    val_fraction: float = 0.2
    deterministic: bool = True
    checkpoint_every: int = 1
    loss: Literal["mse", "gaussian_nll"] = "mse"
    # "auto" -> cuda if available else cpu. MPS is opt-in ("mps") because it is
    # not bit-reproducible.
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    # Validation split strategy, see training.trainer.split_train_val.
    val_strategy: Literal["hash", "random", "newest_round"] = "hash"
    num_dataloader_workers: int = 0


class DistributedConfig(_Section):
    num_workers: int = 2
    use_gpu: bool | Literal["auto"] = "auto"
    cpus_per_worker: int = 1


class EvaluationConfig(_Section):
    max_rmse: float = 0.35
    min_improvement_vs_baseline: float = 0.02
    bootstrap_samples: int = 1000
    mc_samples: int = 30


class ActiveLearningConfig(_Section):
    beta: float = 1.0
    max_cost: float | None = None
    feature_bounds: tuple[float, float] = (-3.0, 3.0)


class TrackingConfig(_Section):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment: str = "bci-closed-loop"
    registered_model: str = "bci-surrogate"


class ServeConfig(_Section):
    num_replicas: int = 1
    max_batch_size: int = 64
    batch_wait_timeout_s: float = 0.01
    port: int = 8000


# Sections that define the science (hashed into ``config_hash``); see
# ``PlatformConfig.scientific_dump``.
SCIENTIFIC_SECTIONS: tuple[str, ...] = (
    "seed",
    "data",
    "model",
    "training",
    "evaluation",
    "active_learning",
)


class PlatformConfig(_Section):
    seed: int = 0
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)

    # ------------------------------------------------------------------ helpers
    def with_overrides(self, **dotted: Any) -> PlatformConfig:
        """Return a copy with dotted-key overrides, e.g. ``training__epochs=3``
        or ``{"training.epochs": 3}`` via ``with_overrides(**{"training.epochs": 3})``."""
        data = self.model_dump(mode="python")
        for key, value in dotted.items():
            parts = key.replace("__", ".").split(".")
            node = data
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
        return PlatformConfig.model_validate(data)

    def scientific_dump(self) -> dict[str, Any]:
        """The part of the config that determines *what* is computed.

        Keeps ``seed``, ``data``, ``model``, ``training``, ``evaluation``,
        ``active_learning`` and ``distributed.num_workers`` (the number of DDP
        replicas sets the global batch size, so it changes the trained model).
        Drops environment/placement details that do not change results:
        ``paths`` (moving the data dir), ``tracking`` (MLflow URI/names),
        ``serve`` and ``distributed.use_gpu/cpus_per_worker``.
        """
        full = self.model_dump(mode="json")
        out = {k: full[k] for k in SCIENTIFIC_SECTIONS}
        out["distributed"] = {"num_workers": full["distributed"]["num_workers"]}
        return out

    def config_hash(self) -> str:
        """Stable hash of :meth:`scientific_dump` (the ``config_hash`` lineage tag)."""
        from bci_platform.hashing import hash_config

        return hash_config(self.scientific_dump())

    def to_flat_dict(self) -> dict[str, Any]:
        """Flattened ``section.key -> value`` mapping (handy for experiment tracking)."""
        out: dict[str, Any] = {}

        def rec(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    rec(f"{prefix}.{k}" if prefix else str(k), v)
            else:
                out[prefix] = obj

        rec("", self.model_dump(mode="json"))
        return out

    @classmethod
    def for_tests(cls, tmp_dir: Path | None = None, **overrides: Any) -> PlatformConfig:
        """A tiny, fast configuration for unit tests."""
        cfg = cls(
            seed=0,
            data=DataConfig(pool_size=2000, initial_observations=300, batch_size_per_round=50),
            model=ModelConfig(hidden_dims=[32, 32], dropout=0.1),
            training=TrainingConfig(epochs=5, batch_size=64, lr=3e-3, device="cpu"),
            distributed=DistributedConfig(num_workers=2, use_gpu=False),
            evaluation=EvaluationConfig(bootstrap_samples=200, mc_samples=10),
        )
        if tmp_dir is not None:
            cfg.paths = PathsConfig(
                data_dir=tmp_dir / "data",
                reports_dir=tmp_dir / "reports",
                artifacts_dir=tmp_dir / "artifacts",
            )
            cfg.tracking.tracking_uri = f"sqlite:///{tmp_dir / 'mlflow.db'}"
        return cfg.with_overrides(**overrides) if overrides else cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a mapping")
    parent = raw.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        raw = _deep_merge(_read_yaml(parent_path), raw)
    return raw


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is None:
        path = os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        # allow bare names such as "distributed" or "distributed.yaml"
        candidate = CONFIG_DIR / (p.name if p.suffix else f"{p.name}.yaml")
        if candidate.exists():
            return candidate
    return p


def load_config(path: str | os.PathLike[str] | None = None) -> PlatformConfig:
    """Load a `PlatformConfig` from YAML (``BCI_CONFIG`` or ``configs/local.yaml``)."""
    p = resolve_config_path(path)
    return PlatformConfig.model_validate(_read_yaml(p))
