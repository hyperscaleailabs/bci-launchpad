"""Checkpoint at epoch 3 -> simulated worker failure -> Ray Train restores -> finish."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import ray

from merge_platform.config import PlatformConfig
from merge_platform.data import (
    generate_candidate_pool,
    initial_observations,
    make_oracle,
    records_to_frame,
)
from merge_platform.inference import Predictor
from merge_platform.ray_runtime.cluster import ensure_ray, shutdown_ray
from merge_platform.ray_runtime.tasks import start_experiment_simulator
from merge_platform.training.distributed import distributed_info, train_distributed

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ray_cluster() -> Iterator[None]:
    already = ray.is_initialized()
    ensure_ray(num_cpus=4, include_dashboard=False)
    yield
    if not already:
        shutdown_ray()


def test_worker_failure_resumes_from_checkpoint(ray_cluster: None, tmp_path: Path) -> None:
    cfg = PlatformConfig.for_tests(tmp_path)
    pool = generate_candidate_pool(cfg)
    frame = records_to_frame(initial_observations(pool, make_oracle(cfg), 300, seed=0))

    result = train_distributed(
        cfg, train_frame=frame, num_workers=2, fail_at_epoch=3, max_failures=1
    )
    info = distributed_info(result)
    assert info["failures_recovered"] == 1
    assert info["restored_from_epoch"] == 3
    assert result.resumed_from is not None and result.resumed_from.endswith("epoch_0003")
    assert result.epochs_completed == cfg.training.epochs
    # history is continuous: epochs 1..3 restored from the checkpoint, 4..N re-run
    assert [int(h["epoch"]) for h in result.history] == list(range(1, cfg.training.epochs + 1))
    assert info["params_in_sync"]
    Predictor.from_checkpoint(result.checkpoint_path)


def test_failure_without_retries_surfaces(ray_cluster: None, tmp_path: Path) -> None:
    cfg = PlatformConfig.for_tests(tmp_path).with_overrides(**{"training.epochs": 3})
    pool = generate_candidate_pool(cfg)
    frame = records_to_frame(initial_observations(pool, make_oracle(cfg), 200, seed=0))
    with pytest.raises(Exception, match=r"(?i)simulated worker failure|TrainingFailed"):
        train_distributed(cfg, train_frame=frame, num_workers=2, fail_at_epoch=2, max_failures=0)


def test_experiment_retry_returns_recorded_measurement(ray_cluster: None, tmp_path: Path) -> None:
    cfg = PlatformConfig.for_tests(tmp_path, **{"data.pool_size": 300})
    pool = generate_candidate_pool(cfg)
    journal = tmp_path / "lab_journal.jsonl"
    sim = start_experiment_simulator(cfg, pool, journal_path=journal)
    ids = pool["candidate_id"].iloc[:8].tolist()
    first: pd.DataFrame = ray.get(sim.measure.remote(1, ids))
    ray.kill(sim)  # the "lab" process dies; a new one reloads the journal
    sim2 = start_experiment_simulator(cfg, pool, journal_path=journal)
    again: pd.DataFrame = ray.get(sim2.measure.remote(1, ids))
    pd.testing.assert_frame_equal(first, again)
    assert ray.get(sim2.stats.remote())["n_physical_measurements"] == 0
