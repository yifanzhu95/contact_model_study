"""Action continuity: raw commands, horizon boundaries, modes and graph replay."""
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
from ContactModelStudy.Simulators.VectorizedSimulator import DeviceState


@wp.kernel
def _broadcast(q: wp.array(dtype=float), v: wp.array(dtype=float),
               positions: wp.array2d(dtype=float), velocities: wp.array2d(dtype=float)):
    n, j = wp.tid()
    positions[n, j] = q[j]
    velocities[n, j] = v[j]


@wp.kernel
def _select(sequence: wp.array3d(dtype=float), t: int, target: wp.array2d(dtype=float)):
    n, j = wp.tid()
    target[n, j] = sequence[n, t, j]


@wp.kernel
def _servo(target: wp.array2d(dtype=float), q: wp.array2d(dtype=float),
           filtered: wp.array2d(dtype=float)):
    n, j = wp.tid()
    # Expose a filtered control deliberately: the rate cost must use raw targets.
    filtered[n, j] = 0.1 * target[n, j]
    q[n, j] = q[n, j] + filtered[n, j]


class Servo:
    """Two-joint deterministic plant with a deliberately distinct filtered control."""
    def __init__(self, horizon=3, device='cpu', n=2):
        self.N, self.nu, self.nq, self.nv, self.horizon = n, 2, 2, 2, horizon
        self.config = SimpleNamespace(device=device, resolved_substeps=1)
        self.state = DeviceState(*(wp.zeros((n, 2), device=device) for _ in range(3)), site_xpos=None)
        self.target = wp.zeros((n, 2), device=device)
        self.sequence = None

    def DeviceState(self):
        return self.state

    def BroadcastState(self, q, v):
        wp.launch(_broadcast, (self.N, 2), inputs=[q,v], outputs=[self.state.qpos,self.state.qvel])

    def SetControl(self, u):
        wp.copy(self.target, u)

    def SetControlSequence(self, sequence):
        self.sequence, self.cursor = sequence, 0

    def ClearControlSequence(self):
        self.sequence = None

    def Step_GPU(self, steps):
        for _ in range(steps):
            if self.sequence is not None:
                wp.launch(_select, (self.N, 2), inputs=[self.sequence,self.cursor], outputs=[self.target])
                self.cursor += 1
            wp.launch(_servo, (self.N, 2), inputs=[self.target], outputs=[self.state.qpos,self.state.ctrl])


class StateCost:
    def calcCosts(self, sim, terminal=False, out=None):
        out.fill_(7.0 if terminal else 2.0)
        return out


def planner(mode='absolute', horizon=3, weight=1., device='cpu', graph=False, n=2):
    return MPPI(Servo(horizon, device, n), StateCost(), MPPI_Config(
        action_rate_weight=weight, control_mode=mode, use_graph=graph,
        seed=123, noise_sigma=.5, temperature=.05))


def score(p, samples, q, previous):
    p._setCommandBase(q, previous)
    p.V_wp.assign(np.asarray(samples, dtype=np.float32))
    p._buildCommands()
    p._rollout(q, np.zeros(2))
    return p.costs_wp.numpy()+p.terminal_costs_wp.numpy()


@pytest.mark.parametrize('mode', ['absolute','ctrl_relative','pos_relative'])
def test_raw_command_cost_matches_independent_moving_plant(mode):
    samples = np.array([[[.2,.1],[-.1,.3],[.4,-.2]],
                        [[-.3,.4],[.5,-.6],[-.2,.7]]])
    q, previous, weight = np.array([1.,-2.]), np.array([.6,-.7]), 2.5
    expected = []
    for tape in samples:
        position, command = q.copy(), previous.copy()
        cost = 0.
        for sampled in tape:
            target = sampled if mode == 'absolute' else sampled + (position if mode == 'pos_relative' else command)
            cost += weight * np.sum((target-command)**2)
            command = target.copy()
            position += .1*target
        expected.append(cost + 2*2 + 7)  # terminal state cost replaces running cost
    with wp.ScopedDevice('cpu'):
        p = planner(mode, weight=weight)
        np.testing.assert_allclose(score(p,samples,q,previous),expected,rtol=1e-6)


@pytest.mark.parametrize('horizon', [1,3])
def test_held_target_is_free_and_first_change_charged_once_even_at_terminal(horizon):
    with wp.ScopedDevice('cpu'):
        p = planner(horizon=horizon,weight=2.)
        tape = np.full((2,horizon,2),.5)
        state_cost = (horizon-1)*2 + 7
        # The joints differ from the held target; neither gravity error nor the
        # backend's filtered control should acquire an action-rate penalty.
        np.testing.assert_allclose(score(p,tape,np.array([-2.,1.]),np.array([.5,.5])),state_cost)
        # A changed previous command must be picked up on the next rollout.
        np.testing.assert_allclose(score(p,tape,np.array([-2.,1.]),np.array([-.5,-.5])),state_cost+4)


def test_penalty_moves_the_optimizer_towards_the_previous_target():
    with wp.ScopedDevice('cpu'):
        previous = np.array([.6,-.6])
        q = np.array([-2.,2.])
        actions = []
        for weight in [0.,10.]:
            p = planner(horizon=2, weight=weight,n=256)
            actions.append(p.Plan(q,np.zeros(2),u=previous))
            assert p.last_plan_ok
        assert np.linalg.norm(actions[1]-previous) < .8*np.linalg.norm(actions[0]-previous)


@pytest.mark.parametrize('mode', ['absolute','pos_relative'])
def test_zero_weight_keeps_original_objective_and_ignores_unused_reference(mode):
    with wp.ScopedDevice('cpu'):
        left, right = planner(mode,weight=0), planner(mode,weight=0)
        a = left.Plan(np.ones(2),np.zeros(2),u=np.array([-10.,10.]))
        b = right.Plan(np.ones(2),np.zeros(2),u=np.array([10.,-10.]))
        np.testing.assert_array_equal(a,b)
        assert left._rate_previous_wp is None


def test_rate_reference_falls_back_to_last_action_and_explicit_u_overrides():
    with wp.ScopedDevice('cpu'):
        p = planner()
        q = np.array([.2,.3])
        p._setCommandBase(q,None)
        np.testing.assert_allclose(p._u_prev_wp.numpy(),q)
        last = p.Plan(q,np.zeros(2),u=np.array([.6,-.6]))
        saved = p.SaveState()
        p._setCommandBase(q,None)
        np.testing.assert_allclose(p._u_prev_wp.numpy(),last)
        p.FreeEnergyOf(q,np.zeros(2),p.last_mean,u=np.array([4.,5.]))
        p.LoadState(saved)
        p._setCommandBase(q,None)
        np.testing.assert_array_equal(p.SaveState()['u_prev'],saved['u_prev'])
        p.Reset()
        p._setCommandBase(q,np.array([.8,.9]))
        np.testing.assert_allclose(p._u_prev_wp.numpy(),[.8,.9])


@pytest.mark.parametrize('weight', [-1.,float('nan'),float('inf')])
def test_invalid_weight_rejected_by_config_and_drivers(weight):
    from ContactModelStudy.Drivers import run_episodes
    import run_leap_hardware
    with pytest.raises(ValueError,match='action_rate_weight'):
        MPPI_Config(action_rate_weight=weight)
    for parse in [run_episodes.parseArgs,run_leap_hardware.parseArgs]:
        with pytest.raises(SystemExit):
            parse([f'--action-rate-weight={weight}'])


def test_cli_defaults_disable_and_logging_configuration():
    from dataclasses import asdict
    from ContactModelStudy.Drivers import run_episodes
    import run_leap_hardware
    assert run_episodes.buildPlannerConfig(run_episodes.parseArgs([])).action_rate_weight == 0
    assert run_episodes.buildPlannerConfig(run_leap_hardware.parseArgs([])).action_rate_weight == 0
    for weight in [0,2.5]:
        args = run_leap_hardware.parseArgs(['--action-rate-weight',str(weight)])
        assert asdict(run_episodes.buildPlannerConfig(args))['action_rate_weight'] == weight


@pytest.mark.gpu
@pytest.mark.parametrize('mode', ['absolute','ctrl_relative','pos_relative'])
def test_cuda_graph_replay_uses_fresh_reference_and_resets_horizon_history(mode):
    with wp.ScopedDevice('cuda:0'):
        eager = planner(mode,device='cuda:0')
        graph = planner(mode,device='cuda:0',graph=True)
        samples = np.array([[[.2,.1],[-.1,.3],[.4,-.2]],
                            [[-.3,.4],[.5,-.6],[-.2,.7]]])
        for previous in [np.array([.6,-.7]),np.array([-.3,.2])]:
            a = score(eager,samples,np.array([1.,-2.]),previous)
            b = score(graph,samples,np.array([1.,-2.]),previous)
            np.testing.assert_allclose(a,b,rtol=1e-6)
        assert graph._rollout_graph is not None and not graph._graph_failed


@pytest.mark.gpu
@pytest.mark.parametrize('contact', ['M1','M2','M3','M4','M6'])
def test_calibrated_backend_executes_action_rate_graph(tmp_path, contact):
    from ContactModelStudy.Drivers import run_episodes
    from ContactModelStudy.Hardware.Workflow import build_tasks, build_rollout
    import run_leap_hardware as hardware
    args = hardware.parseArgs(['--device','cuda:0','--rollout-model',contact,
        '--n-samples','16','--horizon','3','--substeps','1','--action-rate-weight','1'])
    with wp.ScopedDevice('cuda:0'):
        task, evaluation, _ = build_tasks(args,tmp_path)
        sim = build_rollout(args,task,*hardware.model_limits(evaluation.mjm))
        try:
            p = MPPI(sim,task,run_episodes.buildPlannerConfig(args))
            q, v, u = evaluation.getInitialState()
            for reference in [u,u+.02]:
                sim.activation.assign(u.astype(np.float32))
                command = p.Plan(q,v,u=reference)
                assert p.last_plan_ok and np.isfinite(command).all()
            assert p._rollout_graph is not None and not p._graph_failed
            assert p.config.delta_range == (None,None)
        finally:
            sim.Close()
