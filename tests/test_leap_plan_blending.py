"""CPU-only command-history tests; no camera, serial device or GPU is used."""
import numpy as np
import pytest

from ContactModelStudy.Hardware.PlanBlending import TemporalPlanBlend


def tape(values):
    return np.repeat(np.asarray(values, dtype=float)[:, None], 16, axis=1)


def test_blends_the_overlapping_future_slot_not_the_previous_executed_action():
    blend = TemporalPlanBlend(.6, .064)
    old = tape([.15, -.1, .3])
    result, context = blend.prepare(old, old[0], 9.99)
    np.testing.assert_array_equal(result, old[0])
    blend.commit(context, 10.)
    new = tape([.2, .25, -.1])
    result, context = blend.prepare(new, new[0], 10.074)
    np.testing.assert_allclose(result, .6*.2 + .4*(-.1))
    # This would be .18 if it filtered the last executed action (.15).
    assert not np.allclose(result, .18)
    assert context['old_slot'] == 1
    record = blend.metrics(context)
    np.testing.assert_allclose(record['within_plan_diff_l2'], [4*.05, 4*.35])
    np.testing.assert_allclose(record['aligned_plan_diff_l2'], [4*.3, 4*.05])
    assert record['overlap_steps'] == 2
    np.testing.assert_array_equal(old, tape([.15, -.1, .3]))
    np.testing.assert_array_equal(new, tape([.2, .25, -.1]))


@pytest.mark.parametrize('age,slot', [(0.,0),(.063,0),(.064,1),(.130,2)])
def test_alignment_uses_elapsed_publication_time_instead_of_replan_count(age, slot):
    blend = TemporalPlanBlend(0., .064)
    old = tape([.1,.2,.3])
    _, context = blend.prepare(old, old[0], 1.)
    blend.commit(context, 2.)  # Long planning/pacing time before publication.
    result, context = blend.prepare(tape([9.,8.,7.]), np.full(16,9.), 2.+age)
    np.testing.assert_array_equal(result, old[slot])
    assert context['old_slot'] == slot


def test_expiration_reset_and_unpublished_plan_never_reuse_invalid_history():
    blend = TemporalPlanBlend(.2, .064)
    old = tape([.1,.2,.3])
    _, first = blend.prepare(old, old[0], 0.)
    # A prepared but unpublished plan must not become history.
    _, context = blend.prepare(old, old[0], .01)
    assert context['status'] == 'first_plan'
    blend.commit(first, 1.)
    new = tape([.8,.7,.6])
    for now in [1.192, 2.]:
        result, context = blend.prepare(new, new[0], now)
        np.testing.assert_array_equal(result, new[0])
        assert context['status'] == 'expired'
        assert blend.metrics(context)['aligned_plan_diff_l2'] is None
    blend.reset()
    _, context = blend.prepare(new, new[0], 3.)
    assert context['status'] == 'first_plan'


def test_history_is_the_raw_previous_plan_not_a_recursively_smoothed_plan():
    blend = TemporalPlanBlend(0., .064)
    old, new = tape([.1,.2,.3]), tape([.7,.8,.9])
    _, context = blend.prepare(old, old[0], 0.)
    blend.commit(context, 0.)
    result, context = blend.prepare(new, new[0], .064)
    np.testing.assert_allclose(result, .2)
    blend.commit(context, .064)
    result, _ = blend.prepare(tape([4.,5.,6.]), np.full(16,4.), .128)
    np.testing.assert_allclose(result, .8)


def test_alpha_one_is_bit_exact_even_when_recording_metrics():
    blend = TemporalPlanBlend(1., .064)
    rng = np.random.default_rng(17)
    for k in range(10):
        mean = rng.normal(size=(5,16))
        # Plan's resolved-target subtraction can differ slightly from U[0].
        delta = np.nextafter(mean[0], np.inf)
        before = mean.copy()
        result, context = blend.prepare(mean, delta, k*.074)
        np.testing.assert_array_equal(result, delta)
        np.testing.assert_array_equal(mean, before)
        blend.commit(context, k*.074+.001)


@pytest.mark.parametrize('alpha', [-.01,1.01,float('inf'),float('nan')])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match='alpha'):
        TemporalPlanBlend(alpha,.064)


def test_nonfinite_plan_fails_before_it_can_be_published():
    blend = TemporalPlanBlend(.5,.064)
    with pytest.raises(ValueError, match='finite'):
        blend.prepare(tape([.1,float('nan'),.3]),np.zeros(16),0.)


def test_cli_defaults_and_invalid_modes():
    import run_leap_hardware as runner
    assert runner.parseArgs([]).plan_blend_alpha == 1
    assert not runner.parseArgs([]).plan_blend_metrics
    assert runner.parseArgs(['--hardware','--plan-blend-alpha','.6']).plan_blend_alpha == .6
    for args in [['--plan-blend-alpha','.6'], ['--hardware','--observe-only','--plan-blend-alpha','.6'],
                 ['--hardware','--plan-blend-alpha','nan'], ['--hardware','--plan-blend-alpha','1.1'],
                 ['--hardware','--plan-blend-metrics','--no-logging']]:
        with pytest.raises(SystemExit):
            runner.parseArgs(args)
