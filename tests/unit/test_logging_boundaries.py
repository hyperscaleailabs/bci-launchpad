from __future__ import annotations

import json
import subprocess
import sys

from merge_platform.logging import bound_ids, configure_logging, get_logger


def test_json_logging_binds_ids(capsys) -> None:
    configure_logging(fmt="json", level="INFO", force=True)
    try:
        log = get_logger("t", run_id="r1", round_id=None)
        with bound_ids(dataset_id="d1", model_version="3"):
            log.info("hello", x=1)
        line = capsys.readouterr().err.strip().splitlines()[-1]
        rec = json.loads(line)
        assert rec["event"] == "hello" and rec["run_id"] == "r1"
        assert rec["dataset_id"] == "d1" and rec["model_version"] == "3"
        assert "round_id" not in rec
    finally:
        configure_logging(fmt="console", level="WARNING", force=True)


def test_scientific_code_does_not_import_orchestration_frameworks() -> None:
    code = (
        "import sys\n"
        "import merge_platform.data, merge_platform.models, merge_platform.evaluation\n"
        "import merge_platform.training, merge_platform.inference.predictor\n"
        "bad = [m for m in ('ray', 'dagster', 'mlflow') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
