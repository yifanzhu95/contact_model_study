"""CPU-only tests for the Polaris operator package's verdict and manifest."""

from __future__ import annotations

import importlib.util
import json

from conftest import REPO_ROOT


def _module(name):
    path = REPO_ROOT / "experiments" / "polaris" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"polaris_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, document):
    path.write_text(json.dumps(document))


def test_smoke_verdict_requires_completion_validation_and_clean_sha(tmp_path):
    verify = _module("verify_smoke")
    sha = "a" * 40
    _write(tmp_path / "preflight.json", {"passed": True})
    _write(tmp_path / "cell_0000_polaris_g0_m5_cube.status.json", {
        "execution_status": "completed",
        "validation_status": "pass",
        "scaling_eligible": True,
        "provenance": {"source": {"commit": sha, "dirty": False}},
    })
    _write(tmp_path / "cell_0000_polaris_g0_m5_cube.json", {
        "validation": {"status": "pass", "scaling_eligible": True},
        "configs": {"metadata": {"provenance": {
            "source": {"commit": sha, "dirty": False},
        }}},
    })
    assert verify.main(["--outdir", str(tmp_path), "--expected-sha", sha]) == 0
    assert json.loads((tmp_path / "smoke_verdict.json").read_text())["passed"]

    status = json.loads((tmp_path / "cell_0000_polaris_g0_m5_cube.status.json").read_text())
    status["validation_status"] = "fail"
    status["scaling_eligible"] = False
    _write(tmp_path / "cell_0000_polaris_g0_m5_cube.status.json", status)
    assert verify.main(["--outdir", str(tmp_path), "--expected-sha", sha]) == 2
    assert not json.loads((tmp_path / "smoke_verdict.json").read_text())["scaling_eligible"]


def test_artifact_manifest_hashes_outputs_and_records_exit_code(tmp_path):
    collect = _module("collect_manifest")
    (tmp_path / "result.txt").write_text("result\n")
    assert collect.main(["--outdir", str(tmp_path), "--job-exit-code", "2"]) == 0
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["job_exit_code"] == 2
    assert manifest["files"] == [{
        "path": "result.txt",
        "bytes": 7,
        "sha256": "5656fafa00d4f294bcb606cf4f7d4fa877390e46f583e8b3c8744ace104a31d1",
    }]
