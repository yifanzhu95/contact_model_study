"""Regression checks for episode boundaries and fixed-particle diagnostics."""

from types import SimpleNamespace
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest

from analysis.kl_shrinkage_sensitivity import analyze_cell, kl_at_alpha
from contact_study.evaluation import json_io
from contact_study.evaluation.distributions import weighted_moments_from_particles
from experiments.hpc import run_kl_divergence_cell as worker


def test_worker_scientific_defaults_match_cluster_template():
    script = Path(worker.__file__).with_name("kl_divergence_eval.slurm")
    output = subprocess.check_output(["bash", str(script), "--dry-run", "0"], text=True)
    tokens = shlex.split(output.splitlines()[-1])
    explicit = worker.build_parser().parse_args(tokens[2:])
    defaults = worker.build_parser().parse_args([])
    for name in ("model", "geometry", "goal_difficulty", "time_horizon", "step_time",
                 "temperature", "noise_sigma", "kl_every", "kl_shrinkage",
                 "ref_n_samples", "ref_n_iterations", "execute", "settle"):
        assert getattr(defaults, name) == getattr(explicit, name), name


def test_raw_moments_reproduce_kl_without_new_particles():
    rng = np.random.default_rng(71)
    record = {}
    expected = []
    for name, n in (("reference", 100), ("degraded", 16)):
        particles = rng.normal(0, 0.025, (n, 16))
        weights = rng.random(n)
        mu, raw, ess = weighted_moments_from_particles(particles, weights, 0, 0.025)
        _, cov, _ = weighted_moments_from_particles(particles, weights, 0.001, 0.025)
        record[name] = {"mean": mu.tolist(), "covariance_raw": raw.tolist(), "ess": ess}
        expected.append((mu, cov))
    fwd, _ = kl_at_alpha(record, 0.001, 0.025)
    assert fwd == pytest.approx(worker.gaussian_kl(*expected[0], *expected[1]), rel=1e-13)
    assert np.linalg.matrix_rank(record["degraded"]["covariance_raw"]) <= 15
    assert kl_at_alpha(record, 0.0001, 0.025)[0] > kl_at_alpha(record, 0.01, 0.025)[0]


@pytest.mark.parametrize("ending", ["success", "failed", "timeout"])
def test_final_command_is_scored_and_shadow_never_controls(monkeypatch, ending):
    args = worker.build_parser().parse_args(["--max_steps", "1", "--settle", "0",
                                            "--no-record_trajectory", "--no-record_planner_dist"])
    cfg = SimpleNamespace(timestep=0.0005, eval_substeps_per_rollout=8,
                          force_limits=None, control_limits=None, max_steps=1,
                          name="grasp_reorient")
    sim = SimpleNamespace(q=np.array([0.0, 0.0]), v=np.zeros(2), applied=[])
    sim.reset = lambda q, v: None
    sim.get_state = lambda: SimpleNamespace(qpos=sim.q.copy(), qvel=sim.v.copy())
    sim.apply_control = lambda u: sim.applied.append(u.copy())
    def advance(n):
        assert n == 128
        sim.q[:] = 1.0
    sim.step = advance
    mjm = SimpleNamespace(opt=SimpleNamespace(timestep=0), nq=2, nv=2)
    mjd = SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2), ctrl=np.zeros(2))
    task = SimpleNamespace(config=cfg, load=lambda: (mjm, mjd),
                           get_inital_state=lambda rng: (np.zeros(2), np.zeros(2), np.zeros(2)),
                           make_eval_simulator=lambda **kw: sim,
                           is_success=lambda data: ending == "success" and data.qpos[0] == 1,
                           has_failed=lambda data: ending == "failed" and data.qpos[0] == 1)
    monkeypatch.setattr(worker, "get_task", lambda *a, **kw: task)
    monkeypatch.setattr(worker, "apply_goal_difficulty", lambda *a: None)
    monkeypatch.setattr(worker.mujoco, "mj_forward", lambda *a: None)
    planners = []
    class Buffer:
        def __init__(self): self.value = np.ones((2, 2)) * 0.03
        def numpy(self): return self.value.copy()
        def assign(self, x): self.value[:] = x
    class Planner:
        def __init__(self, **kw):
            self.horizon, self.nu, self.substeps = 2, 2, 16
            self.control_dt, self.robot_qpos_adr = 0.064, 0
            self.pc, self.U_wp = SimpleNamespace(ctrl_relative_to_qpos=True), Buffer()
            self.last_plan_ok = True
            self.ref = len(planners) == 1
            planners.append(self)
        def plan(self, data):
            np.testing.assert_array_equal(data.qpos, [0, 0])
            # The shadow must receive pre-solve U0, not the degraded solution.
            np.testing.assert_array_equal(self.U_wp.value, np.full((2, 2), 0.03))
            self.U_wp.value[:] = 9 if self.ref else 0.1
            return np.full(2, 9 if self.ref else 0.1)
    monkeypatch.setattr(worker, "MPPIController", Planner)
    monkeypatch.setattr(worker, "weighted_moments", lambda *a: (np.zeros(2), np.eye(2), 2))
    planner_cfg = SimpleNamespace(noise_sigma=0.025, n_samples=16)
    result, record = worker.run_kl_episode(args, SimpleNamespace(label="M3"),
                                         planner_cfg, planner_cfg, np.random.default_rng(0),
                                         "cube_high_high", None, 0)
    assert result.end_reason == ending
    assert result.success == (ending == "success")
    assert result.steps_to_success == (1 if ending == "success" else None)
    assert result.n_steps_taken == 1
    np.testing.assert_array_equal(sim.applied, [[0.1, 0.1]])
    assert record["steps"] == [0]


def test_optional_moment_recording_preserves_estimator():
    class Buffer:
        def __init__(self, value): self.value = value
        def numpy(self): return self.value
    rng = np.random.default_rng(42)
    controller = SimpleNamespace(V_wp=Buffer(rng.normal(size=(16, 5, 16))),
                                 w_wp=Buffer(rng.random(16)))
    args = worker.build_parser().parse_args([])
    standard = worker.measure_distribution(controller, args, 0.025)
    args.record_kl_moments = True
    recorded = worker.measure_distribution(controller, args, 0.025)
    for left, right in zip(standard[:3], recorded[:3]):
        np.testing.assert_array_equal(left, right)
    payload = json_io.dumps(recorded[3], precision=0)
    assert '"covariance_raw"' in payload


def test_zero_raw_covariance_has_known_regularized_kl():
    record = {"reference": {"mean": [0.0], "covariance_raw": [[0.0]]},
              "degraded": {"mean": [1.0], "covariance_raw": [[0.0]]}}
    # Both variances are alpha, so KL is (1 - 0)^2 / (2 * alpha).
    assert kl_at_alpha(record, 0.1, 1.0) == pytest.approx((5.0, 5.0))
    assert kl_at_alpha(record, 0.01, 1.0) == pytest.approx((50.0, 50.0))
    for alpha in (0, -1, float("nan"), float("inf"), 2):
        with pytest.raises(ValueError):
            kl_at_alpha(record, alpha, 1.0)


def test_saved_kl_must_match_recorded_moments(tmp_path):
    record = {"step": 0,
              "reference": {"mean": [0.0], "covariance_raw": [[0.0]], "ess": 1},
              "degraded": {"mean": [1.0], "covariance_raw": [[0.0]], "ess": 1}}
    payload = {"config": {"record_kl_moments": True, "noise_sigma": 1, "kl_shrinkage": 0.1},
               "per_step": [{"steps": [0], "moments": [record], "kl_forward": [5.0], "kl_reverse": [5.0]}]}
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(payload))
    cell = {"source_paths": [str(path)], "label": "test", "geometry": "cube_high_high",
            "family_id": "test", "null": False}
    report = analyze_cell(cell, [0.01, 0.1])
    assert [point["mean"] for point in report["curve"]] == pytest.approx([50, 5])
    assert report["saved_kl_reproduction_abs_error"]["max"] < 1e-12
    payload["per_step"][0]["kl_forward"][0] = 7.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="do not reproduce"):
        analyze_cell(cell, [0.1])


def test_compact_plotting_example_explicitly_omits_raw_moments(tmp_path):
    path = tmp_path / "compact.json"
    path.write_text(json.dumps({"config": {"record_kl_moments": True},
                               "artifact": {"omitted_fields": ["per_step.moments"]}}))
    cell = {"source_paths": [str(path)], "label": "test", "geometry": "cube_high_high",
            "family_id": "test", "null": False}
    report = analyze_cell(cell, [0.001])
    assert report["files_without_moments"] == 1
    assert report["measurements"] == 0
