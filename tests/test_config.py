"""tests/test_config.py

Unit tests for ContactModelConfig and the dispatch layer.
These tests do NOT require a GPU — they test the config logic,
factory methods, and dispatch routing in isolation using mocks.
"""

import pytest
import numpy as np

from contact_study.contact_models.config import (
    Backend,
    ContactModelConfig,
    GeometryVariant,
    ComfreeParams,
    XPBDParams,
    PhysicsNoiseParams,
    MujocoSolverParams,
)


# ---------------------------------------------------------------------------
# Factory method tests
# ---------------------------------------------------------------------------

class TestFactoryMethods:
    def test_M1_backend(self):
        cfg = ContactModelConfig.M1()
        assert cfg.backend == Backend.MUJOCO_ANITESCU
        assert cfg.mujoco.cone == "pyramidal"
        assert cfg.mujoco.solver == "Newton"

    def test_M2_backend(self):
        cfg = ContactModelConfig.M2()
        assert cfg.backend == Backend.MUJOCO_SOFT
        assert cfg.mujoco.cone == "elliptic"

    def test_M3_backend(self):
        cfg = ContactModelConfig.M3()
        assert cfg.backend == Backend.COMFREE

    def test_M4_backend(self):
        cfg = ContactModelConfig.M4()
        assert cfg.backend == Backend.XPBD
        assert cfg.xpbd.damping_friction is False

    def test_M4_damping_friction_variant(self):
        cfg = ContactModelConfig.M4(damping_friction=True)
        assert cfg.xpbd.damping_friction is True
        assert "dampfric" in cfg.label

    def test_M5_geometry_variant(self):
        cfg = ContactModelConfig.M5(GeometryVariant.PRIMITIVE_UNION)
        assert cfg.geometry == GeometryVariant.PRIMITIVE_UNION
        assert cfg.backend == Backend.MUJOCO_SOFT

    def test_M7_physics_noise(self):
        cfg = ContactModelConfig.M7(friction_sigma=0.3)
        assert cfg.physics_noise.friction_sigma == pytest.approx(0.3)
        assert cfg.backend == Backend.MUJOCO_SOFT

    def test_M10_combined(self):
        cfg = ContactModelConfig.M10(
            geom=GeometryVariant.CONVEX_HULL, friction_sigma=0.2
        )
        assert cfg.backend == Backend.XPBD
        assert cfg.geometry == GeometryVariant.CONVEX_HULL
        assert cfg.physics_noise.friction_sigma == pytest.approx(0.2)

    def test_all_models_returns_list(self):
        models = ContactModelConfig.all_models()
        assert len(models) >= 10
        # All should have unique labels
        labels = [m.label for m in models]
        assert len(labels) == len(set(labels)), "Duplicate labels in all_models()"

    def test_default_label_generated(self):
        cfg = ContactModelConfig(backend=Backend.COMFREE)
        assert cfg.label is not None
        assert len(cfg.label) > 0


# ---------------------------------------------------------------------------
# Parameter block defaults
# ---------------------------------------------------------------------------

class TestParamDefaults:
    def test_comfree_defaults(self):
        p = ComfreeParams()
        assert p.stiffness == pytest.approx(0.2)
        assert p.damping == pytest.approx(0.001)

    def test_xpbd_defaults(self):
        p = XPBDParams()
        assert p.compliance == pytest.approx(1e-4)
        assert p.iterations == 5
        assert p.damping_friction is False

    def test_physics_noise_zero_by_default(self):
        p = PhysicsNoiseParams()
        assert p.mass_sigma == 0.0
        assert p.friction_sigma == 0.0
        assert p.com_sigma == 0.0

    def test_mujoco_solver_defaults(self):
        p = MujocoSolverParams()
        assert p.cone == "elliptic"
        assert p.solver == "PGS"


# ---------------------------------------------------------------------------
# Physics noise application (no GPU needed)
# ---------------------------------------------------------------------------

class TestPhysicsNoise:
    def test_zero_noise_returns_same_values(self):
        """_apply_physics_noise with all sigma=0 should not change model values."""
        try:
            import mujoco
        except ImportError:
            pytest.skip("mujoco not installed")

        import tempfile, os
        xml = """
        <mujoco>
          <worldbody>
            <body name="b" pos="0 0 0">
              <freejoint/>
              <geom type="sphere" size="0.05" mass="1.0"/>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml)
            path = f.name

        try:
            mjm = mujoco.MjModel.from_xml_path(path)
            orig_mass = mjm.body_mass.copy()

            from contact_study.contact_models.api import _apply_physics_noise
            rng = np.random.default_rng(0)
            noise = PhysicsNoiseParams()  # all zeros
            mjm2 = _apply_physics_noise(mjm, noise, rng)

            np.testing.assert_array_almost_equal(mjm2.body_mass, orig_mass)
        finally:
            os.unlink(path)

    def test_nonzero_noise_changes_mass(self):
        try:
            import mujoco
        except ImportError:
            pytest.skip("mujoco not installed")

        import tempfile, os
        xml = """
        <mujoco>
          <worldbody>
            <body name="b" pos="0 0 0">
              <freejoint/>
              <geom type="sphere" size="0.05" mass="1.0"/>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml)
            path = f.name

        try:
            mjm = mujoco.MjModel.from_xml_path(path)
            orig_mass = mjm.body_mass.copy()

            from contact_study.contact_models.api import _apply_physics_noise
            rng = np.random.default_rng(42)
            noise = PhysicsNoiseParams(mass_sigma=0.5)
            mjm2 = _apply_physics_noise(mjm, noise, rng)

            # At least one body mass should have changed
            assert not np.allclose(mjm2.body_mass, orig_mass), \
                "Mass should have been perturbed with sigma=0.5"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Geometry variant
# ---------------------------------------------------------------------------

class TestGeometryVariant:
    def test_all_variants_have_values(self):
        for v in GeometryVariant:
            assert isinstance(v.value, str)
            assert len(v.value) > 0

    def test_xml_path_template_format(self):
        """XML template string should be formattable with geometry value."""
        from contact_study.tasks.base import TaskSpec, ContactComplexity
        spec = TaskSpec(
            name="push",
            complexity=ContactComplexity.LOW,
            xml_path_template="tasks/push_{geometry}.xml",
            max_steps=100,
            success_threshold=0.02,
        )
        for geom in GeometryVariant:
            path = spec.xml_path_template.format(geometry=geom.value)
            assert "{" not in path, f"Unformatted template for {geom}: {path}"


if __name__=="__main__":
    a = TestFactoryMethods()
    a.test_M1_backend()