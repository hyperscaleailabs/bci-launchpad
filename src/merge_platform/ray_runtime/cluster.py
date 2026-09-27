"""Ray cluster connection.

The single place that decides *which* Ray cluster the platform talks to.
Everything else calls :func:`ensure_ray` and stays agnostic of whether it is a
local in-process cluster (laptop), a Docker Compose head node, or KubeRay.

Resolution order: explicit ``address`` argument, then ``RAY_ADDRESS`` env var,
otherwise start a local cluster.
"""

from __future__ import annotations

import os
from typing import Any

import ray

from merge_platform.config import REPO_ROOT
from merge_platform.logging import get_logger

log = get_logger(__name__)

# Workers must be able to import `merge_platform` and resolve repo-relative
# paths regardless of where Ray starts them.
_RUNTIME_ENV: dict[str, Any] = {
    "env_vars": {
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        # Ray Train V2's controller polls workers every 2 s by default and a
        # worker's ``ray.train.report`` waits for that poll, which puts a ~2 s
        # floor under every epoch of our small models. Poll faster.
        "RAY_TRAIN_HEALTH_CHECK_INTERVAL_S": os.environ.get(
            "RAY_TRAIN_HEALTH_CHECK_INTERVAL_S", "0.2"
        ),
    }
}


def _disable_uv_run_hook() -> None:
    """Stop Ray from rebuilding a fresh uv venv for every worker under ``uv run``.

    Ray >= 2.4x detects a driver started via ``uv run`` and, by default,
    packages the working dir and re-runs ``uv`` in each worker's runtime env
    (~10 s per new worker process, plus a copy of the repo in /tmp). Workers
    here share the driver's interpreter/venv, so that is pure overhead. An
    explicit ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` in the environment is respected.
    """
    if "RAY_ENABLE_UV_RUN_RUNTIME_ENV" in os.environ:
        return
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    try:
        from ray._private import ray_constants

        ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False  # read at import time
    except (ImportError, AttributeError):  # pragma: no cover - future Ray versions
        pass


def ensure_ray(
    address: str | None = None,
    *,
    num_cpus: int | None = None,
    namespace: str = "merge_platform",
    **init_kwargs: Any,
) -> dict[str, Any]:
    """Initialise Ray once per process (idempotent) and return cluster resources."""
    if not ray.is_initialized():
        _disable_uv_run_hook()
        address = address or os.environ.get("RAY_ADDRESS") or None
        kwargs: dict[str, Any] = {
            "namespace": namespace,
            "ignore_reinit_error": True,
            "log_to_driver": True,
            "runtime_env": _RUNTIME_ENV,
            **init_kwargs,
        }
        if address:
            kwargs["address"] = address
        elif num_cpus is not None:
            kwargs["num_cpus"] = num_cpus
        ctx = ray.init(**kwargs)
        log.info(
            "ray_initialized",
            address=address or "local",
            dashboard_url=getattr(ctx, "dashboard_url", None),
        )
    return dict(ray.cluster_resources())


def shutdown_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()
