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
    }
}


def ensure_ray(
    address: str | None = None,
    *,
    num_cpus: int | None = None,
    namespace: str = "merge_platform",
    **init_kwargs: Any,
) -> dict[str, Any]:
    """Initialise Ray once per process (idempotent) and return cluster resources."""
    if not ray.is_initialized():
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
