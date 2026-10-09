"""CPU math, inverse-fit, checkpoint, and renderer/validation integration checks.

The rasterizer is a differentiable CPU stand-in; CUDA needs a server smoke run.
"""
import ast
import argparse
from contextlib import redirect_stdout
import io
import importlib.util
import math
import os
import random
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if "--require_torch" in sys.argv:
    sys.argv.remove("--require_torch")
    if not TORCH_AVAILABLE:
        raise SystemExit("SH preflight requires Torch; use the physir environment")
if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F
    from utils.sh_utils import eval_sh
    from utils.thermal_sh_residual import ThermalSHResidual, thermal_view_radiance, sh2_basis
    from utils.thermal_physics import save_thermal_checkpoint
    from utils.thermal_training_utils import thermal_data_loss


def source_function(relative, name, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class FreshStage2Tests(unittest.TestCase):
    def test_runner_reuses_settings_but_never_Stage2_weights(self):
        from tools.run_sh_residual import build_training_command
        args = SimpleNamespace(output="new", radiometric_dir="", steps=30000, degree=2,
            start_step=3000, bound=0.5, lr=0.01, lr_final=0.0001, regularization=0.001, lambda_tv=None)
        shared = {"lambda_tv": 0.003, "temperature_lr": 1e-4, "loss_scale_mode": "fit_quantile"}
        metadata = {"shared_training_protocol": shared, "geometry_iteration": 30000, "seed": 17}
        config = {"temp_min": 250, "temp_max": 450, "temp_ref": 300}
        plans = []
        for degree in (0, 2):
            args.degree = degree
            command = build_training_command(args, metadata, config, ROOT, "scene", "geometry", "masks", "materials")
            self.assertNotIn("--stage2_checkpoint", command)
            self.assertEqual(command[command.index("--stage") + 1], "stage2")
            self.assertEqual(command[command.index("--lambda_tv") + 1], "0.003")
            plans.append(command)
        index = plans[0].index("--sh_residual_degree") + 1
        plans[0][index] = plans[1][index]
        self.assertEqual(*plans)


@unittest.skipUnless(TORCH_AVAILABLE, "Torch unavailable")
class SHTorchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.xyz = torch.randn(10, 3)
        self.support = torch.arange(10) < 8

    def test_basis_matches_existing_3DGS_SH_without_DC_or_offset(self):
        directions = F.normalize(torch.randn(40, 3), dim=-1)
        coefficients = torch.randn(40, 8)
        full = torch.cat((torch.zeros(40, 1), coefficients), dim=-1)[:, None, :]
        expected = eval_sh(2, full, directions)[:, 0]
        self.assertTrue(torch.allclose(expected, (coefficients * sh2_basis(directions)).sum(-1), atol=1e-6))

    def test_addition_theorem_no_DC_and_global_bound(self):
        directions = F.normalize(torch.randn(2000, 3), dim=-1)
        basis = sh2_basis(directions)
        self.assertTrue(torch.allclose(basis.square().sum(-1), torch.full((2000,), 8 / (4 * math.pi)), atol=5e-7))
        axes = torch.cat((torch.eye(3), -torch.eye(3)))
        self.assertTrue(torch.allclose(sh2_basis(axes).mean(0), torch.zeros(8), atol=1e-7))
        field = ThermalSHResidual(torch.zeros(2000, 3), torch.ones(2000, dtype=torch.bool), 0.04, True)
        with torch.no_grad():
            field.raw.normal_(0, 100)
        value = field(directions, torch.zeros(3))
        self.assertLessEqual(float(value.detach().abs().max()), 0.040001)

    def test_zero_initial_identity_and_unseen_gaussians(self):
        field = ThermalSHResidual(self.xyz, self.support, 0.05, True)
        physical = torch.linspace(0, 1, 10)[:, None]
        baseline, _, _ = thermal_view_radiance(physical, self.xyz, torch.zeros(3))
        actual, delta, clipped = thermal_view_radiance(physical, self.xyz, torch.zeros(3), field)
        self.assertTrue(torch.equal(actual, baseline))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertEqual(float(clipped), 0)
        with torch.no_grad():
            field.raw.fill_(1)
        self.assertTrue(torch.equal(field(self.xyz, torch.zeros(3))[~self.support], torch.zeros(2, 1)))

    def test_clipping_and_joint_gradients_keep_geometry_frozen(self):
        xyz = self.xyz.clone().requires_grad_()
        field = ThermalSHResidual(xyz, self.support, 0.5, True)
        physical = torch.full((10, 1), 0.5, requires_grad=True)
        prediction, _, _ = thermal_view_radiance(physical, xyz, torch.zeros(3), field)
        (prediction.square().mean() + 0.001 * field.regularizer()).backward()
        self.assertIsNone(xyz.grad)
        self.assertTrue(bool((physical.grad != 0).all()))
        self.assertTrue(bool((field.raw.grad[self.support].abs().sum(-1) > 0).all()))
        self.assertTrue(bool((field.raw.grad[~self.support] == 0).all()))
        with torch.no_grad():
            field.raw[:, 0] = 100
        prediction, delta, clipped = thermal_view_radiance(torch.zeros(10, 1), xyz, torch.zeros(3), field)
        self.assertTrue(bool(((prediction >= 0) & (prediction <= 1)).all()))
        self.assertTrue(bool((delta < 0).any()))
        self.assertGreater(float(clipped), 0)

    def test_fits_angular_signal_where_constant_baseline_cannot(self):
        # Same Gaussian, different cameras, known smooth directional signal.
        xyz = torch.zeros(1, 3)
        cameras = F.normalize(torch.randn(32, 3), dim=-1)
        target = 0.5 + 0.06 * cameras[:, 0] - 0.04 * cameras[:, 1] * cameras[:, 2]
        field = ThermalSHResidual(xyz, [True], 0.2, True)
        optimizer = torch.optim.Adam([field.raw], lr=0.03)
        baseline_mse = float((target - target.mean()).square().mean())
        for _ in range(220):
            prediction = torch.stack([thermal_view_radiance(torch.full((1, 1), 0.5), xyz, c, field)[0][0, 0]
                                      for c in cameras])
            loss = (prediction - target).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertLess(float(loss.detach()), baseline_mse * 0.001)

    def test_checkpoint_roundtrip_old_baseline_and_frozen_branch(self):
        field = ThermalSHResidual(self.xyz, self.support, 0.04, True)
        with torch.no_grad():
            field.raw.normal_()
        model = SimpleNamespace(export_state=lambda: {"branch": "stage2", "state_dict": {}})
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "thermal_stage2_common.pt")
            save_thermal_checkpoint(path, model, 23, {"example": True}, sh_residual=field)
            checkpoint = torch.load(path, map_location="cpu")
        restored = ThermalSHResidual.from_checkpoint(checkpoint, self.xyz, False)
        self.assertFalse(restored.raw.requires_grad)
        self.assertTrue(torch.equal(restored(self.xyz, torch.zeros(3)), field(self.xyz, torch.zeros(3))))
        self.assertIsNone(ThermalSHResidual.from_checkpoint({}, self.xyz))
        with self.assertRaises((ValueError, RuntimeError)):
            ThermalSHResidual.from_checkpoint(checkpoint, self.xyz[:5])
        checkpoint["sh_residual"]["degree"] = 3
        with self.assertRaises(ValueError):
            ThermalSHResidual.from_checkpoint(checkpoint, self.xyz)

    def test_actual_renderer_preserves_SH_gradients_with_detached_geometry(self):
        class Rasterizer:
            def __init__(self, raster_settings):
                self.settings = raster_settings
            def __call__(self, **kwargs):
                color = (kwargs["colors_precomp"] * kwargs["opacities"]).sum(0)
                color = color + 0.01 * kwargs["means3D"].sum()
                return color[:, None, None].expand(3, 2, 3), torch.ones(10), torch.zeros(1)
        namespace = {"torch": torch, "math": math, "GaussianModel": object,
            "GaussianRasterizationSettings": SimpleNamespace, "GaussianRasterizer": Rasterizer}
        render = source_function("gaussian_renderer/__init__.py", "render", namespace)
        xyz = self.xyz.clone().requires_grad_()
        opacity = torch.full((10, 1), 0.4, requires_grad=True)
        pc = SimpleNamespace(get_xyz=xyz, get_opacity=opacity, get_scaling=torch.ones(10, 3),
            get_rotation=torch.ones(10, 4), get_features=torch.zeros(10, 1, 3), _features_dc=torch.zeros(10, 1, 3), active_sh_degree=0)
        camera = SimpleNamespace(image_height=2, image_width=3, FoVx=1., FoVy=1.,
            world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        field = ThermalSHResidual(xyz, self.support, 0.05, True)
        colors, _, _ = thermal_view_radiance(torch.full((10, 1), 0.2), xyz, camera.camera_center, field)
        output = render(camera, pc, pipe, torch.zeros(3), 0., 0., 0., "cpu",
            override_color=colors.repeat(1, 3), detach_geometry=True)["render"]
        output.square().mean().backward()
        self.assertGreater(float(field.raw.grad.abs().sum()), 0)
        self.assertIsNone(xyz.grad)
        self.assertIsNone(opacity.grad)

    def test_training_validation_evaluates_each_camera_direction(self):
        def render(camera, gaussians, pipe, background, *a, **kwargs):
            return {"render": kwargs["override_color"].mean(0)[:, None, None]}
        namespace = {"torch": torch, "math": math, "F": F, "render": render,
            "thermal_view_radiance": thermal_view_radiance, "render_opacity": lambda x: None,
            "thermal_data_loss": thermal_data_loss, "observation_to_model_radiance": lambda x, *a: x}
        validate = source_function("train_thermal_physics.py", "validation_metrics", namespace)
        xyz = torch.zeros(1, 3)
        field = ThermalSHResidual(xyz, [True], 0.2, True)
        with torch.no_grad():
            field.raw[0, 2] = 0.5
        views = []
        for center in (torch.tensor([1., 0, 0]), torch.tensor([-1., 0, 0])):
            target = thermal_view_radiance(torch.full((1, 1), 0.5), xyz, center, field)[0]
            views.append(SimpleNamespace(camera_center=center, original_physical_image=target[:, None], original_image=None))
        physical = SimpleNamespace(radiance=lambda lut: torch.full((1, 1), 0.5))
        args = SimpleNamespace(observation_domain="normalized_dn", huber_delta=0.02)
        baseline = validate(physical, None, views, SimpleNamespace(get_xyz=xyz), None, torch.zeros(3), SimpleNamespace(is_6dof=False), args)
        corrected = validate(physical, None, views, SimpleNamespace(get_xyz=xyz), None, torch.zeros(3), SimpleNamespace(is_6dof=False), args, sh_field=field)
        self.assertGreater(baseline["signal_RMSE"], 0.01)
        self.assertEqual(corrected["signal_RMSE"], 0)

    def test_invalid_configuration(self):
        for bound in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ThermalSHResidual(self.xyz, self.support, bound, True)
        with self.assertRaises(ValueError):
            ThermalSHResidual(self.xyz, [True], 0.1, True)
        with self.assertRaises(ValueError):
            ThermalSHResidual(self.xyz, torch.zeros(10, dtype=torch.bool), 0.1, True)

    def test_fresh_training_loop_saves_warmup_final_and_selected_states(self):
        # Execute the real training loop on a small scene; only scene loading
        # and CUDA rasterization are replaced. Exercises optimizer scheduling,
        # checkpoint selection and field initialization together.
        import numpy as np
        from unittest.mock import patch
        from arguments import ModelParams, PipelineParams
        from utils.thermal_physics import (MaterialThermalField, UniformLWIRPlanckLUT,
            build_spatial_tv_edges, frozen_geometry_state)
        from utils.thermal_opacity import IROpacityCorrection, render_opacity
        from utils.thermal_training_utils import held_lr, install_loss_scale, PlateauTracker, quality_status

        class Gaussians:
            def __init__(self, *args):
                self.active_sh_degree = self.max_sh_degree = 3
                for name, shape in (("_xyz", (2, 3)), ("_features_dc", (2, 1, 3)),
                    ("_features_rest", (2, 15, 3)), ("_thermal_features_dc", (2, 1, 3)),
                    ("_thermal_features_rest", (2, 15, 3)), ("_opacity", (2, 1)),
                    ("_scaling", (2, 3)), ("_rotation", (2, 4))):
                    setattr(self, name, torch.nn.Parameter(torch.ones(shape)))
                self._xyz.data[:, 0] = torch.tensor([-0.1, 0.1])
            @property
            def get_xyz(self):
                return self._xyz
            @property
            def get_opacity(self):
                return torch.sigmoid(self._opacity)

        class Scene:
            def __init__(self, dataset, gaussians, **kwargs):
                self.gaussians, self.loaded_iter, self.has_rgbt = gaussians, 30000, True
                self.views = [SimpleNamespace(image_name=str(i), camera_center=torch.tensor([float(i - 2), 0., 0.]),
                    original_physical_image=torch.full((3, 2, 2), 0.46 + 0.02 * i), original_image=None)
                    for i in range(5)]
            def getTrainCameras(self):
                return self.views[:4]
            def getTestCameras(self):
                return self.views[4:]

        def render(camera, gaussians, pipe, background, *a, **kwargs):
            image = kwargs["override_color"].mean(0)[:, None, None].expand(3, 2, 2)
            return {"render": image, "visibility_filter": torch.ones(2, dtype=torch.bool), "radii": torch.ones(2)}

        config = {"names": ["paint"], "epsilon0": [0.95], "unknown_label": 255, "unknown_epsilon0": 0.95,
            "learn_k": [True], "learn_delta": [True], "k_prior": [0.], "sigma_k": [1e-4],
            "sigma_delta": [0.05], "temperature_reference_K": 300., "document": {"example": True}}
        namespace = dict(torch=torch, np=np, F=F, math=math, os=os, json=__import__("json"),
            random=random, argparse=argparse, ModelParams=ModelParams, PipelineParams=PipelineParams,
            GaussianModel=Gaussians, Scene=Scene, render=render, MaterialThermalField=MaterialThermalField,
            IROpacityCorrection=IROpacityCorrection, render_opacity=render_opacity,
            ThermalSHResidual=ThermalSHResidual, thermal_view_radiance=thermal_view_radiance,
            build_spatial_tv_edges=build_spatial_tv_edges, frozen_geometry_state=frozen_geometry_state,
            save_thermal_checkpoint=save_thermal_checkpoint, held_lr=held_lr, install_loss_scale=install_loss_scale,
            thermal_data_loss=thermal_data_loss, PlateauTracker=PlateauTracker, quality_status=quality_status,
            file_sha256=lambda p: "example", geometry_checkpoint_sha256=lambda *a: "example",
            load_material_config=lambda p: config, load_temperature_truth=lambda *a: None,
            prepare_thermal_input=lambda args, *a: UniformLWIRPlanckLUT(args.temp_min, args.temp_max, num_temp=128),
            map_material_masks_to_gaussians=lambda *a, **k: (torch.zeros(2, dtype=torch.long), torch.ones(2)),
            map_thermal_observations_to_gaussians=lambda *a, **k: torch.full((2, 1), 0.5))
        for name in ("parse_args", "observation_to_model_radiance", "split_fit_validation_cameras",
                     "apply_stage2_protocol", "apply_regularization_protocol", "enforce_comparison_protocol",
                     "checkpoint_metadata", "make_field", "validation_metrics", "main"):
            source_function("train_thermal_physics.py", name, namespace)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new_stage2"
            argv = ["train", "--stage", "stage2", "--geometry_model", tmp, "-s", tmp, "-m", str(output),
                "--material_config", "example.json", "--steps", "6", "--save_every", "2", "--eval_every", "2",
                "--sh_residual_degree", "2", "--sh_residual_start_step", "3", "--gradient_log_every", "2"]
            # This scene/rasterizer is a CPU stand-in even on CUDA servers.
            # The real parser must not move the field to cuda:0 while the
            # fake geometry, labels and visibility tensors remain on CPU.
            # The patch ends before the runner starts real GPU training.
            with patch.object(torch.cuda, "is_available", return_value=False), \
                    patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                namespace["main"]()
            warmup = torch.load(output / "thermal_stage2_step_2.pt", map_location="cpu")
            final = torch.load(output / "thermal_stage2_final.pt", map_location="cpu")
            common = torch.load(output / "thermal_stage2_common.pt", map_location="cpu")
            self.assertTrue(torch.equal(warmup["sh_residual"]["state_dict"]["raw"], torch.zeros(2, 8)))
            self.assertGreater(float(final["sh_residual"]["state_dict"]["raw"].abs().sum()), 0)
            self.assertIsNone(final["metadata"]["warm_start_checkpoint_sha256"])
            self.assertTrue(common["metadata"]["stage2_common_endpoint"])
            self.assertEqual(common["metadata"]["sh_residual_protocol"]["degree"], 2)
            self.assertEqual(final["metadata"]["camera_split"], common["metadata"]["camera_split"])
            for name in ("_xyz", "_opacity", "_scaling", "_rotation"):
                self.assertTrue(torch.equal(final["frozen_geometry"][name], common["frozen_geometry"][name]))

    def test_CPU_smoke_remains_on_CPU_when_host_reports_CUDA(self):
        from unittest.mock import patch
        # Reproduce the server's capability detection without requiring a GPU
        # locally. The inner smoke fixture must override this to stay on CPU.
        with patch.object(torch.cuda, "is_available", return_value=True):
            self.test_fresh_training_loop_saves_warmup_final_and_selected_states()
            self.assertTrue(torch.cuda.is_available())


if __name__ == "__main__":
    unittest.main(verbosity=2)
