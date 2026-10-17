"""Regression checks for D handoff, branch equality and frozen-state safety."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from utils.thermal_joint import GEOMETRY_NAMES, geometry_digest
from utils.thermal_physics import MaterialThermalField, UniformLWIRPlanckLUT
from utils.thermal_sh_residual import ThermalSHResidual
from utils.thermal_stage3 import (assert_frozen, export_common, lock_protocol,
                                  restore_models, validate_common)


class Stage3Test(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        field = MaterialThermalField(torch.tensor([0, 0, 1, -1]),
            torch.tensor([[.1], [.2], [.3], [.4]]), [.9, .3],
            material_names=["plastic", "metal"], learn_k_by_material=[True, False],
            learn_delta_by_material=[True, False], initial_environment=.2)
        geometry = {"_xyz": torch.randn(4, 3), "_rotation": torch.randn(4, 4),
                    "_scaling": torch.randn(4, 3), "_opacity": torch.randn(4, 1),
                    "_features_dc": torch.randn(4, 1, 3), "_features_rest": torch.randn(4, 15, 3),
                    "active_sh_degree": 3}
        sh = ThermalSHResidual(geometry["_xyz"], torch.ones(4, dtype=torch.bool), .01)
        sh.raw.data.normal_()
        self.original = field.export_state()
        self.original.update(step=19000, joint_geometry=geometry, sh_residual=sh.export_state(), metadata={
            "source_path": str(self.root), "shared_training_protocol": {"observation_domain": "raw_rjpeg", "lambda_tv": 0},
            "joint_protocol": {"kind": "fixed_count_rgb_ir_D_v1", "gaussian_count": 4,
                               "trained_geometry_sha256": geometry_digest(geometry)}})
        self.source = self.root / "D.pt"
        torch.save(self.original, self.source)
        self.common = export_common(self.original, self.source, ["plastic"])

    def tearDown(self):
        self.directory.cleanup()

    def gaussian(self):
        geometry = self.common["joint_geometry"]
        gaussian = SimpleNamespace(max_sh_degree=3, active_sh_degree=3,
                                   _thermal_features_dc=torch.nn.Parameter(torch.zeros(4, 1, 3)),
                                   _thermal_features_rest=torch.nn.Parameter(torch.zeros(4, 15, 3)))
        for name in GEOMETRY_NAMES:
            setattr(gaussian, name, torch.nn.Parameter(geometry[name].clone() + .1))
        # Real GaussianModel exposes this property; use a compatible subclass.
        class Gaussian(SimpleNamespace):
            @property
            def get_xyz(self):
                return self._xyz
        return Gaussian(**vars(gaussian))

    def test_export_preserves_all_trained_tensors(self):
        for key, value in self.original["state_dict"].items():
            self.assertTrue(torch.equal(value, self.common["state_dict"][key]), key)
        self.assertEqual(geometry_digest(self.original["joint_geometry"]),
                         geometry_digest(self.common["joint_geometry"]))
        self.assertFalse(self.common["metadata"]["stage2_common_endpoint"])

    def test_restore_trained_geometry_instead_of_initializer(self):
        gaussian = self.gaussian()
        field, sh = restore_models(self.common, gaussian, "K", torch.device("cpu"))
        assert_frozen(self.common, field, gaussian, sh)
        self.assertFalse(gaussian._opacity.requires_grad)
        self.assertFalse(gaussian._features_rest.requires_grad)
        self.assertTrue(field.k_raw.requires_grad)
        self.assertFalse(field.delta_epsilon_raw.requires_grad)

    def test_zero_R_K_equal_C_including_unknown(self):
        lut = UniformLWIRPlanckLUT(num_temp=128, num_lambda=32)
        radiances = []
        for branch in ("C", "R", "K"):
            field, _ = restore_models(self.common, self.gaussian(), branch, torch.device("cpu"))
            radiances.append(field.radiance(lut))
        self.assertTrue(torch.equal(radiances[0], radiances[1]))
        self.assertTrue(torch.equal(radiances[0], radiances[2]))

    def test_only_enabled_material_updates_unknown_epsilon_fixed(self):
        lut = UniformLWIRPlanckLUT(num_temp=128, num_lambda=32)
        for branch in ("K", "R"):
            field, _ = restore_models(self.common, self.gaussian(), branch, torch.device("cpu"))
            raw = field.k_raw if branch == "K" else field.delta_epsilon_raw
            initial_unknown = field.emissivity()[-1].detach().clone()
            field.radiance(lut).sum().backward()
            self.assertNotEqual(float(raw.grad[0]), 0)
            self.assertEqual(float(raw.grad[1]), 0)
            torch.optim.Adam([raw], lr=.01).step()
            self.assertNotEqual(float(raw[0]), 0)
            self.assertEqual(float(raw[1]), 0)
            self.assertTrue(torch.equal(initial_unknown, field.emissivity()[-1]))
            self.assertFalse(field.epsilon0_by_material.requires_grad)

    def test_corrupted_common_or_SH24_rejected(self):
        for mutate in (lambda p: p["joint_geometry"]["_opacity"].add_(.1),
                       lambda p: p["state_dict"]["temperature_raw"].add_(.1),
                       lambda p: p["sh_residual"].update(kind="rgb_sh24")):
            payload = copy.deepcopy(self.common)
            mutate(payload)
            with self.assertRaises(ValueError):
                validate_common(payload)

    def test_freeze_guard_detects_geometry_SH_and_buffer_edits(self):
        for target in ("geometry", "sh", "epsilon"):
            gaussian = self.gaussian()
            field, sh = restore_models(self.common, gaussian, "C", torch.device("cpu"))
            with torch.no_grad():
                if target == "geometry":
                    gaussian._xyz.add_(.01)
                elif target == "sh":
                    sh.raw.add_(.01)
                else:
                    field.epsilon0_by_material.add_(.01)
            with self.assertRaises(RuntimeError):
                assert_frozen(self.common, field, gaussian, sh)

    def test_comparison_lock_rejects_different_seed_or_budget(self):
        path = self.root / "lock.json"
        lock_protocol(path, {"seed": 0, "steps": 56})
        lock_protocol(path, {"seed": 0, "steps": 56})
        with self.assertRaises(ValueError):
            lock_protocol(path, {"seed": 1, "steps": 56})
        with self.assertRaises(ValueError):
            lock_protocol(path, {"seed": 0, "steps": 10000})

    def test_GT_supervised_checkpoint_rejected(self):
        payload = copy.deepcopy(self.common)
        payload["metadata"]["shared_training_protocol"]["temperature_gt"] = "truth.npy"
        with self.assertRaises(ValueError):
            validate_common(payload)


if __name__ == "__main__":
    unittest.main()
