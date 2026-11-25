"""Registration startup fixes without a camera, network, motors, or CUDA."""
from types import SimpleNamespace

import numpy as np
import pytest

from ContactModelStudy.Hardware import VisionRuntime as runtime


@pytest.mark.parametrize('n_sample', [None, 1, 19, 10000])
def test_diameter_matches_upstream_sampling_and_preserves_rng(n_sample):
    pts = np.random.default_rng(7).normal(size=(83, 3))
    state = np.random.get_state()
    try:
        np.random.seed(123)
        sample = pts if n_sample is None else pts[
            np.random.choice(len(pts), size=min(n_sample, len(pts)), replace=False)]
        expected = np.linalg.norm(sample[None] - sample[:, None], axis=-1).max()
        expected_next = np.random.random()
        np.random.seed(123)
        actual = runtime.sampled_mesh_diameter(pts, n_sample, block_size=11)
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert np.random.random() == expected_next
    finally:
        np.random.set_state(state)


def test_full_ten_thousand_point_diameter_has_bounded_workspace(monkeypatch):
    import scipy.spatial.distance
    cdist = scipy.spatial.distance.cdist
    shapes = []

    def bounded(a, b, **kwargs):
        shapes.append((len(a), len(b)))
        assert len(a) <= 512 and len(b) <= 512
        return cdist(a, b, **kwargs)

    monkeypatch.setattr(scipy.spatial.distance, 'cdist', bounded)
    pts = np.zeros((10000, 3))
    pts[:, 0] = np.linspace(-2, 3, len(pts))
    assert runtime.sampled_mesh_diameter(pts, n_sample=None) == 5
    assert len(shapes) > 1


def test_adapter_changes_the_estimator_copy_and_preserves_mesh_branch():
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return 123.0

    estimator = SimpleNamespace(compute_mesh_diameter=original)
    runtime.install_bounded_mesh_diameter(estimator)
    installed = estimator.compute_mesh_diameter
    runtime.install_bounded_mesh_diameter(estimator)
    assert estimator.compute_mesh_diameter is installed
    assert installed(model_pts=np.array([[0, 0, 0], [3, 4, 0]])) == 5
    assert not calls
    mesh = object()
    assert installed(mesh=mesh, n_sample=19) == 123
    assert calls == [dict(model_pts=None, mesh=mesh, n_sample=19)]


def test_failed_startup_stage_cancels_stack_timer_and_keeps_exception(monkeypatch, capsys):
    events = []
    monkeypatch.setattr(runtime.faulthandler, 'dump_traceback_later',
                        lambda *a, **kw: events.append('armed'))
    monkeypatch.setattr(runtime.faulthandler, 'cancel_dump_traceback_later',
                        lambda: events.append('cancelled'))
    with pytest.raises(RuntimeError, match='registration failed'):
        with runtime.startup_stage('registration'):
            raise RuntimeError('registration failed')
    assert events == ['armed', 'cancelled']
    output = capsys.readouterr().out
    assert 'registration...' in output and 'ready in' not in output


def test_tracker_pose_projection_preserves_mesh_origin_and_next_frame_cache():
    from scipy.spatial.transform import Rotation
    from ContactModelStudy.Hardware.LCM import tracked_transform
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.2,-.3,.4]).as_matrix()@np.diag([1.00001,.9999,1.00003]);pose[:3,3]=[.01,.02,.6]
    center=np.eye(4);center[:3,3]=[.017,-.023,.009]
    corrected,cache=runtime.normalize_tracker_pose(pose,center)
    np.testing.assert_array_equal(corrected[:3,3],pose[:3,3])
    np.testing.assert_allclose(cache@center,corrected,atol=1e-14)
    np.testing.assert_allclose(tracked_transform(corrected),corrected,atol=1e-14)


def test_tracker_cache_does_not_accumulate_float32_scale_over_repeated_updates():
    center=np.eye(4);center[:3,3]=[.01,-.02,.03];cache=np.eye(4,dtype=np.float32);cache[2,3]=.6
    increment=np.diag([1.,1.,.999998]).astype(np.float32)
    for _ in range(1500):
        cache[:3,:3]=increment@cache[:3,:3]
        corrected,cache=runtime.normalize_tracker_pose(cache@center,center)
        cache=cache.astype(np.float32)
    np.testing.assert_allclose(corrected[:3,:3].T@corrected[:3,:3],np.eye(3),atol=1e-12)


@pytest.mark.parametrize('kind',['scale','reflection','nan'])
def test_tracker_projection_keeps_invalid_pose_checks(kind):
    pose=np.eye(4)
    pose[0,0]={'scale':1.02,'reflection':-1.,'nan':np.nan}[kind]
    with pytest.raises(ValueError):runtime.normalize_tracker_pose(pose,np.eye(4))
