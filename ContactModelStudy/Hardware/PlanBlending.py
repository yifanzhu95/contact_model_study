"""Optional two-plan blending in MPPI's incremental command space.

Rows are zero-order-held commands of duration ``period``. Anchor each old
sequence to its actual controller publication time, not its planning start or
an assumed one-row-per-replan clock. This aligns command slots approximately;
it does not predict state evolution during planning or serial transport.
"""
import math

import numpy as np


class TemporalPlanBlend:
    def __init__(self, alpha, period):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('plan blend alpha must be finite and between 0 and 1')
        if not np.isfinite(period) or period <= 0:
            raise ValueError('plan blend period must be finite and positive')
        self.alpha, self.period = float(alpha), float(period)
        self.reset()

    def reset(self):
        self.previous = None
        self.published_at = None

    def prepare(self, mean, delta, now):
        """Blend row zero, retaining the raw new plan for the next comparison.

        Caller commits only after successfully publishing. Neither input nor
        planner state is modified. An expired plan contributes no action.
        """
        mean, delta = np.asarray(mean, dtype=float), np.asarray(delta, dtype=float)
        if (mean.ndim != 2 or mean.shape[0] < 1 or delta.ndim != 1
                or mean.shape[1] != delta.size or not np.isfinite(mean).all()
                or not np.isfinite(delta).all() or not np.isfinite(now)):
            raise ValueError('plan blending requires a finite (horizon, joints) mean and delta')
        mean = mean.copy()
        # Resolve the tiny rounding difference between Plan's returned target
        # minus its base and U[0]; the first row must match the transmitted delta.
        mean[0] = delta
        age = None if self.published_at is None else float(now - self.published_at)
        if age is not None and age < 0:
            raise ValueError('plan blend clock moved backwards')
        slot = None if age is None else math.floor((age + 1e-10) / self.period)
        status, aligned = 'first_plan', None
        if self.previous is not None:
            if self.previous.shape[1] != delta.size:
                raise ValueError('plan blend joint count changed without reset')
            if slot >= len(self.previous):
                status = 'expired'
            else:
                status = 'overlap'
                count = min(len(mean), len(self.previous) - slot)
                aligned = self.previous[slot:slot + count]
        result = delta.copy()
        if aligned is not None and self.alpha != 1:
            result = self.alpha * delta + (1 - self.alpha) * aligned[0]
        context = dict(mean=mean, aligned=aligned, result=result.copy(), status=status,
                       age_s=age, old_slot=slot, prepared_at=float(now))
        return result, context

    def commit(self, context, published_at):
        if not np.isfinite(published_at) or published_at < context['prepared_at']:
            raise ValueError('plan publication must follow blend preparation')
        self.previous, self.published_at = context['mean'], float(published_at)

    def metrics(self, context):
        """Post-publication diagnostics, all in delta radians (not velocities)."""
        mean, old = context['mean'], context['aligned']
        within = np.diff(mean, axis=0)
        between = None if old is None else mean[:len(old)] - old
        norms = lambda a: np.linalg.norm(a, axis=1).tolist()
        return dict(alpha=self.alpha, space='incremental_joint_delta',
                    alignment='elapsed_publication_time_zero_order_hold',
                    status=context['status'], age_s=context['age_s'], old_slot=context['old_slot'],
                    overlap_steps=0 if old is None else len(old),
                    new_mean=mean.tolist(),
                    within_plan_diff_l2=norms(within),
                    aligned_plan_diff_l2=None if between is None else norms(between),
                    first_aligned_disagreement=None if between is None else between[0].tolist(),
                    old_aligned_delta=None if old is None else old[0].tolist(),
                    unblended_delta=mean[0].tolist(), blended_delta=context['result'].tolist(),
                    blend_adjustment_l2=float(np.linalg.norm(context['result'] - mean[0])))
