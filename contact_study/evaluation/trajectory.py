"""Per-control-step recording of an eval episode: state, control, planner belief.

What this exists for:

  replay          `steps.ctrl` is the exact absolute command handed to
                  sim.apply_control() each control step, and `steps.qpos` /
                  `steps.qvel` are the eval state it was computed from. Together
                  with `context` they satisfy the identity

                      u_k = qpos[k][adr : adr+nu] + action[k]     (servo param.)
                      u_k = (u_{k-1} if k else u0) + action[k]    (accumulating)
                      u_k = clip(u_k, clip_lo, clip_hi)
                      u_k == ctrl[k]

                  so an episode can be re-driven open-loop, or its actions fed to
                  a different simulator, without re-running the planner.

  KL divergence   `planner_dist` records the planner's induced first-action
                  distribution (mean sequence, weighted mean, full covariance,
                  ESS) at each recorded step. A reference planner replayed on the
                  recorded states yields the second distribution, and
                  evaluation/distributions.gaussian_kl closes the loop — offline,
                  without the shadow solve that run_kl_divergence_cell.py needs.

Two things deliberately NOT recorded:

  the settle phase   The drivers hold u0 for settle_seconds before the control
                     loop, with no planner involved. Step 0's state IS the
                     settled state, which is all a replay needs. Note this puts
                     trajectory time and VIDEO time settle_seconds apart: the
                     eval sims capture frames on their own clock inside step(),
                     so the recording opens with the settle and the trajectory
                     does not. context.settle_seconds carries the offset.

  the KL cell's reference planner   Its shadow solve is not part of the episode.

Cost: the distribution pulls V_wp off the GPU, which copies the whole (N, H, nu)
block and syncs the device — 1.8 MB and a full pipeline stall per recorded step
at n_samples=4096. Hence planner_dist_every, and hence the rule that every hook
site calls into this module OUTSIDE the region it is timing with perf_counter.
In the async driver that is a correctness requirement, not a performance one:
that loop spends the measured plan_ms as simulated seconds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from contact_study.evaluation import json_io
from contact_study.evaluation.distributions import planner_moments


__all__ = ["TrajectoryConfig", "TrajectoryRecorder", "add_cli_flags"]

SCHEMA = 1


@dataclass
class TrajectoryConfig:
    """What to record. Both blocks default ON; the sweeps turn them off."""
    record_trajectory:   bool  = True
    record_planner_dist: bool  = True
    # plan() calls between recorded distributions. 1 = every step.
    planner_dist_every:  int   = 1
    # Significant digits floats are rounded to on the way out. 7 is float32's
    # precision, so the planner arrays are lossless; qpos/qvel/ctrl are float64
    # and DO lose digits, so a bit-exact open-loop replay wants precision=0.
    precision:           int   = 7
    # Covariance shrink toward sigma^2 I, matching --kl_shrinkage. 0 keeps the
    # raw weighted covariance, which can be singular when the weights collapse.
    shrinkage:           float = 0.0

    @classmethod
    def from_args(cls, args) -> "TrajectoryConfig":
        """Read the flags off an argparse namespace, tolerating their absence.

        precision and shrinkage have no CLI flag anywhere yet; they are read the
        same tolerant way so adding one later needs no change here.
        """
        d = cls()
        return cls(
            record_trajectory   = bool(getattr(args, "record_trajectory",   d.record_trajectory)),
            record_planner_dist = bool(getattr(args, "record_planner_dist", d.record_planner_dist)),
            planner_dist_every  = max(1, int(getattr(args, "planner_dist_every", d.planner_dist_every))),
            precision           = int(getattr(args, "record_precision", d.precision)),
            shrinkage           = float(getattr(args, "record_shrinkage", d.shrinkage)),
        )

    @property
    def any(self) -> bool:
        return self.record_trajectory or self.record_planner_dist


def add_cli_flags(p) -> None:
    """Add the recording flags to an argparse parser (same three everywhere)."""
    p.add_argument("--record_trajectory", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Record per-control-step state + applied control into the "
                        "result JSON (default on; sweeps pass --no-record_trajectory).")
    p.add_argument("--record_planner_dist", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Also record the planner's first-action distribution "
                        "(mean sequence, weighted mean, covariance, ESS). Pulls "
                        "V_wp off the GPU, so decimate it on long episodes.")
    p.add_argument("--planner_dist_every", type=int, default=1,
                   help="Record the planner distribution every K plans (default 1).")


# ---------------------------------------------------------------------------
# columnar accumulator
# ---------------------------------------------------------------------------
class _Columns:
    """Struct-of-arrays accumulator: one growable column per named field.

    Columnar rather than a list of per-step dicts so each field lands as one
    compact JSON line, the key names are not repeated once per step, and a
    consumer gets np.array(block["qpos"]) in one call.

    Row assignment into a preallocated array copies, which is also what keeps the
    async missed-tick path honest: there `u` is literally the previous tick's
    object, and holding a reference would retroactively rewrite earlier rows.
    """

    def __init__(self, capacity: int):
        self._cap  = max(int(capacity), 1)
        self._n    = 0
        self._cols: dict[str, np.ndarray] = {}

    def append(self, **fields) -> None:
        if self._n == self._cap:
            self._grow()
        for name, val in fields.items():
            col = self._cols.get(name)
            if col is None:
                col = self._alloc(val)
                self._cols[name] = col
            col[self._n] = val
        self._n += 1

    def _alloc(self, val) -> np.ndarray:
        if isinstance(val, (bool, np.bool_)):
            return np.zeros(self._cap, dtype=bool)
        if isinstance(val, (int, np.integer)):
            return np.zeros(self._cap, dtype=np.int64)
        arr = np.asarray(val, dtype=np.float64)
        return np.full((self._cap,) + arr.shape, np.nan, dtype=np.float64)

    def _grow(self) -> None:
        self._cap *= 2
        for name, col in self._cols.items():
            pad = np.zeros((self._cap - col.shape[0],) + col.shape[1:], dtype=col.dtype)
            if col.dtype.kind == "f":
                pad.fill(np.nan)
            self._cols[name] = np.concatenate([col, pad], axis=0)

    def __len__(self) -> int:
        return self._n

    def to_dict(self, precision: int) -> dict:
        return {
            name: json_io.compact(col[: self._n], precision=precision)
            for name, col in self._cols.items()
        }


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------
class TrajectoryRecorder:
    """Accumulates one episode's trajectory. Inert when nothing is enabled.

    Hangs off EpisodeResult.trajectory, so every writer that already serializes
    the episode list picks the block up unchanged.

    Three loops feed it. The synchronous driver and the KL cell call step() once
    per control step. The async driver has no such alignment — plans and executor
    ticks are separate events — so it calls tick() per executor tick and
    plan_event() per completed plan(), and the distribution is keyed by plan
    index rather than by step.
    """

    def __init__(self, cfg: TrajectoryConfig | None, controller, *,
                 driver: str,
                 control_dt: float, rollout_dt: float, eval_dt: float,
                 eval_substeps: int, max_steps: int,
                 clip: tuple = (None, None),
                 settle_seconds: float = 0.0,
                 extra_context: dict | None = None):
        self.cfg        = cfg if cfg is not None else TrajectoryConfig()
        self.controller = controller
        self.driver     = driver
        self.enabled    = self.cfg.any

        # In the async driver the distribution is per PLAN, not per control step.
        self._dist_key  = "plan_idx" if driver == "async" else "step"

        self._steps = _Columns(max_steps)
        self._plans = _Columns(max(max_steps // 4, 8))
        self._dist  = _Columns(max(max_steps // max(self.cfg.planner_dist_every, 1), 8))
        self._dist_kind: str | None = None
        self._goal_switch_steps: list[int] = []
        self._action_source = "mean"

        clip_lo, clip_hi = clip
        nu = int(getattr(controller, "nu", 0))
        pc = getattr(controller, "pc", None)
        self._nu = nu
        self.context = {
            "nu":                        nu,
            "horizon":                   int(getattr(controller, "horizon", 0)),
            "substeps":                  int(getattr(controller, "substeps", 0)),
            "robot_qpos_adr":            int(getattr(controller, "robot_qpos_adr", 0)),
            "ctrl_relative_to_qpos":     bool(getattr(pc, "ctrl_relative_to_qpos", True)),
            "n_samples":                 int(getattr(pc, "n_samples", 0)),
            "n_iterations":              int(getattr(pc, "n_iterations", 1)),
            # MPPI-only; None on planners that always run a fixed count.
            "convergence_tol":           getattr(pc, "convergence_tol", None),
            "max_iterations":            int(getattr(pc, "max_iterations", 0)),
            "resample_per_iteration":    bool(getattr(pc, "resample_per_iteration", False)),
            "noise_sigma":               float(getattr(pc, "noise_sigma", 0.0)),
            "warm_start":                bool(getattr(pc, "warm_start", False)),
            "time_constrained":          bool(getattr(pc, "time_constrained", False)),
            "control_dt":                float(control_dt),
            "rollout_dt":                float(rollout_dt),
            "eval_dt":                   float(eval_dt),
            "eval_substeps_per_rollout": int(eval_substeps),
            "eval_steps_per_control":    int(getattr(controller, "substeps", 0)) * int(eval_substeps),
            "max_steps":                 int(max_steps),
            "clip_lo":                   None if clip_lo is None else float(np.min(clip_lo)),
            "clip_hi":                   None if clip_hi is None else float(np.max(clip_hi)),
            # The settle runs BEFORE step 0 and is not recorded; the video does
            # include it, so video time leads trajectory time by this much.
            "settle_seconds":            float(settle_seconds),
            "n_settle_steps":            int(settle_seconds / rollout_dt) if rollout_dt else 0,
            "planner_dist_every":        int(self.cfg.planner_dist_every),
            "precision":                 int(self.cfg.precision),
            "shrinkage":                 float(self.cfg.shrinkage),
        }
        if extra_context:
            self.context.update(extra_context)

    # -- synchronous / KL hook ---------------------------------------------
    def step(self, *, step: int, t: float, qpos, qvel, action, ctrl,
             plan_ms: float, action_source: str = "mean") -> None:
        """One control step, called AFTER the clip and BEFORE apply_control()."""
        if not self.enabled:
            return
        self._action_source = action_source
        if self.cfg.record_trajectory:
            self._steps.append(
                step=int(step), t=float(t),
                qpos=np.asarray(qpos, dtype=np.float64),
                qvel=np.asarray(qvel, dtype=np.float64),
                action=self._as_action(action),
                ctrl=np.asarray(ctrl, dtype=np.float64),
                plan_ms=float(plan_ms),
            )
        self._maybe_dist(int(step))

    # -- asynchronous hooks -------------------------------------------------
    def plan_event(self, *, plan_idx: int, t_start: float, t_visible: float,
                   plan_ms: float, latency_ms: float) -> None:
        """One completed plan(). MUST be called outside the timed region."""
        if not self.enabled:
            return
        if self.cfg.record_trajectory:
            self._plans.append(
                plan_idx=int(plan_idx), t_start=float(t_start),
                t_visible=float(t_visible), plan_ms=float(plan_ms),
                latency_ms=float(latency_ms),
            )
        self._maybe_dist(int(plan_idx))

    def tick(self, *, step: int, t: float, qpos, qvel, action, ctrl,
             tape_id: int, tape_row: int, staleness_ms: float | None,
             applied: bool) -> None:
        """One executor tick, missed ticks included (applied=False, action=None)."""
        if not self.enabled or not self.cfg.record_trajectory:
            return
        self._steps.append(
            step=int(step), t=float(t),
            qpos=np.asarray(qpos, dtype=np.float64),
            qvel=np.asarray(qvel, dtype=np.float64),
            action=self._as_action(action),
            ctrl=np.asarray(ctrl, dtype=np.float64),
            tape_id=int(tape_id), tape_row=int(tape_row),
            staleness_ms=float("nan") if staleness_ms is None else float(staleness_ms),
            applied=bool(applied),
        )

    # -- episode structure --------------------------------------------------
    def goal_switch(self, step: int) -> None:
        """Multi-goal mode resampled the goal here and reset the planner.

        steps_to_success only records the FIRST success, so these are what let a
        consumer segment the episode. The reset zeroes the planner's mean, so the
        mean_seq immediately after a switch is a fresh solve and looks
        discontinuous; in the async driver the tape is voided too, which shows up
        as a run of applied=False ticks.
        """
        if self.enabled:
            self._goal_switch_steps.append(int(step))

    def finish(self) -> dict | None:
        """The trajectory block, or None when recording was off."""
        if not self.enabled:
            return None
        prec = self.cfg.precision
        # Vector-valued context entries (q0/v0/u0, and anything a driver adds)
        # get the same one-line treatment as the step columns, so the header
        # stays scannable instead of running one scalar per line.
        ctx = {k: json_io.compact(v, precision=prec)
                  if isinstance(v, (np.ndarray, list, tuple)) else v
               for k, v in self.context.items()}
        ctx["action_source"]     = self._action_source
        ctx["goal_switch_steps"] = json_io.compact(self._goal_switch_steps)

        out: dict = {"schema": SCHEMA, "driver": self.driver, "context": ctx}
        if self.cfg.record_trajectory and len(self._steps):
            out["steps"] = self._steps.to_dict(prec)
        if self.driver == "async" and self.cfg.record_trajectory and len(self._plans):
            out["plans"] = self._plans.to_dict(prec)
        if self.cfg.record_planner_dist and len(self._dist):
            out["planner_dist"] = {
                "kind": self._dist_kind or "unknown",
                # Keyed by control step (sync/KL) or by plans.plan_idx (async).
                "key":  self._dist_key,
                **self._dist.to_dict(prec),
            }
        return out

    # -- internals ----------------------------------------------------------
    def _as_action(self, action):
        """A missed async tick applies no delta; keep the column rectangular."""
        if action is None:
            return np.zeros(self._nu, dtype=np.float64)
        return np.asarray(action, dtype=np.float64)

    def _maybe_dist(self, index: int) -> None:
        if not self.cfg.record_planner_dist:
            return
        if index % self.cfg.planner_dist_every:
            return

        m = planner_moments(self.controller, shrinkage=self.cfg.shrinkage)
        if self._dist_kind is None:
            self._dist_kind = m.kind

        nu  = self._nu
        # last_action_seq, NOT U_wp: _extract_action stashes the mean pre-shift
        # and then shifts U_wp in place under warm_start, which would leave the
        # recorded sequence off by shift_steps rows from the action beside it.
        # It is already a host copy, so reading it costs nothing.
        seq = getattr(self.controller, "last_action_seq", None)
        H   = int(self.context.get("horizon", 0)) or 1
        self._dist.append(**{
            self._dist_key:    int(index),
            "mean_seq":        np.full((H, nu), np.nan) if seq is None
                               else np.asarray(seq, dtype=np.float64),
            "mu":              np.full(nu, np.nan) if m.mu is None
                               else np.asarray(m.mu, dtype=np.float64),
            "cov":             np.full((nu, nu), np.nan) if m.cov is None
                               else np.asarray(m.cov, dtype=np.float64),
            "ess":             float("nan") if m.ess is None else float(m.ess),
            "n_particles":     int(m.n_particles or 0),
            # < horizon only when the time-constrained path truncated the unroll;
            # mean_seq rows past this were never touched by the update.
            "n_steps_planned": int(getattr(self.controller, "last_n_steps", H)),
            # Optimizer iterations the plan actually ran; below the configured
            # cap only when MPPI's convergence_tol terminated the loop early.
            "n_iterations_run": int(getattr(self.controller, "last_n_iterations", 0)),
            "degenerate":      bool(m.degenerate),
        })
