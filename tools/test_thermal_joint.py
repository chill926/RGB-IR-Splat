"""CPU preflight for D: real losses, gradients, checkpoint and loop integration.

The CUDA rasterizer is replaced only inside smoke fixtures, never production.
"""
import ast
import argparse
import copy
from contextlib import redirect_stdout
import io
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AVAILABLE = importlib.util.find_spec("torch") is not None
if "--require_torch" in sys.argv:
    sys.argv.remove("--require_torch")
    if not AVAILABLE:
        raise SystemExit("D preflight needs Torch in the physir environment")
if AVAILABLE:
    import torch
    import torch.nn.functional as F
    from utils.flir_radiometry import blackbody_signal
    from utils.thermal_joint import (TorchFrozenDisplay, geometry_digest, install_joint_geometry,
        image_loss, joint_signal, masked_ssim, sequential_joint_step)
    from utils.thermal_pseudocolor import FrozenDisplay
    from utils.thermal_physics import MaterialThermalField, UniformLWIRPlanckLUT, frozen_geometry_state
    from utils.thermal_sh_residual import ThermalSHResidual
    from utils.thermal_training_utils import held_lr, install_loss_scale, masked_mean, thermal_data_loss


def source_function(relative, name, namespace, cpu_smoke=False):
    path = ROOT / relative
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name)
    if cpu_smoke == "evaluation":
        assert name == "main"
        node.body[0].value = ast.parse("torch.device('cpu')").body[0].value
        node.body = [item for item in node.body if not (isinstance(item, ast.If) and "is_available" in ast.dump(item.test))]
    elif cpu_smoke:
        # Only bypass the production CUDA capability guard in this CPU fixture.
        assert name == "run" and isinstance(node.body[0], ast.If)
        node.body = node.body[1:]
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def display_fixture(path):
    calibration = dict(PlanckR1=21106.77, PlanckR2=.012545258, PlanckB=1501., PlanckF=1., PlanckO=-7340.)
    q0, q1 = blackbody_signal([250., 450.], calibration)
    settings = {"palette_sha256": "example", "signal_window_center": float((q0 + q1) / 2), "raw_value_range": float(q1 - q0)}
    document = {"format_version": 1, "kind": "flir_train_calibrated_pseudocolor", "signal_calibration": calibration,
        "manifest_sha256": "manifest", "fit_camera_names": ["fit"],
        "groups": {"example": {"palette_rgb": [[.1, .2, .3], [.5, .8, .4], [.9, .1, .7]],
            "response": {"window_knots": [.1, .3, .6, .9], "palette_fraction_knots": [.05, .25, .5, .95],
                "tail_slopes": [.7, 1.7], "support": [.1, .9]}}},
        "frame_display_metadata": {"fit": settings, "validation": dict(settings, raw_value_range=float((q1-q0) * .3))}}
    path.write_text(json.dumps(document))
    return FrozenDisplay(path)


@unittest.skipUnless(AVAILABLE, "Torch unavailable")
class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_display_matches_numpy_inside_outside_support_and_narrow_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen = display_fixture(Path(tmp) / "display.json")
            module = TorchFrozenDisplay(frozen, [250, 450], "cpu")
            values = torch.linspace(0, 1, 231).reshape(1, 11, 21)
            for name in ("fit", "validation"):
                expected, _ = frozen.colorize(values.numpy(), name, [250, 450])
                actual = module(values, name).permute(1, 2, 0).numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-7)
            self.assertEqual(list(module.parameters()), [])

    def test_display_signal_gradient_matches_finite_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen = display_fixture(Path(tmp) / "display.json")
            module = TorchFrozenDisplay(frozen, [250, 450], "cpu")
            value = torch.tensor([[[.43]]], dtype=torch.float64, requires_grad=True)
            module(value, "fit").sum().backward()
            numerical = (module(value.detach() + 1e-6, "fit").sum() - module(value.detach() - 1e-6, "fit").sum()) / 2e-6
            self.assertAlmostEqual(float(value.grad), float(numerical), places=6)
            self.assertGreater(abs(float(value.grad)), .01)

    def test_invalid_native_pixels_do_not_affect_L1_or_structural_loss(self):
        target = torch.rand(3, 25, 27)
        mask = torch.ones(1, 25, 27, dtype=torch.bool)
        mask[:, :3, :] = False
        prediction = target.clone().requires_grad_()
        altered = prediction.detach().clone()
        altered[:, :3, :] = 100
        self.assertAlmostEqual(float(image_loss(altered, target, mask)), 0., places=6)
        image_loss(prediction, target, mask).backward()
        self.assertEqual(float(prediction.grad[:, :3, :].abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, "No valid"):
            masked_ssim(target, target, torch.zeros_like(mask))

    def test_sequential_backwards_match_one_joint_update(self):
        value = torch.nn.Parameter(torch.tensor([.2, .7]))
        expected = value.detach().clone().requires_grad_()
        total = (expected - .4).square().sum() + (expected * 2 - .1).square().sum()
        total.backward()
        optimizer = torch.optim.SGD([{"params": [value], "name": "shared"}], lr=.01)
        sequential_joint_step(optimizer, lambda: (value - .4).square().sum(), lambda: (value * 2 - .1).square().sum())
        torch.testing.assert_close(value.detach(), expected.detach() - .01 * expected.grad)
        # RGB is intentionally absent during the frozen warmup.
        sequential_joint_step(optimizer, lambda: torch.tensor(0.), lambda: value.square().sum())

    def test_SH_direction_updates_xyz_only_when_explicitly_enabled(self):
        xyz = torch.tensor([[.3, .2, .9]], requires_grad=True)
        sh = ThermalSHResidual(xyz, [True], .2, True)
        with torch.no_grad():
            sh.raw[0, 0] = .3
        sh(xyz, torch.zeros(3)).sum().backward()
        self.assertIsNone(xyz.grad)
        sh(xyz, torch.zeros(3), detach_geometry=False).sum().backward()
        self.assertGreater(float(xyz.grad.abs().sum()), 0)

    def test_CPU_images_and_mask_can_initialize_CUDA_geometry(self):
        if not torch.cuda.is_available():
            self.skipTest("Mixed-device initializer check runs on the CUDA server")
        from utils.thermal_physics import map_thermal_observations_to_gaussians
        xyz = torch.tensor([[.2, .3, .7], [.4, .1, .8]], device="cuda")
        pc = SimpleNamespace(get_xyz=xyz, get_opacity=torch.ones(2, 1, device="cuda") * .2)
        camera = SimpleNamespace(original_physical_image=torch.full((3, 20, 20), .2), original_image=None,
            thermal_valid_mask=torch.ones(1, 20, 20, dtype=torch.bool), full_proj_transform=torch.eye(4, device="cuda"),
            image_width=20, image_height=20)
        observed = map_thermal_observations_to_gaussians(pc, [camera], fallback_radiance=.1)
        torch.testing.assert_close(observed.cpu(), torch.full((2, 1), .2))

    def test_restore_trained_geometry_and_reject_tampering_before_mutation(self):
        pc = self.toy_gaussians()
        original = frozen_geometry_state(pc)
        updated = copy.deepcopy(original)
        updated["_xyz"] += .2
        payload = {"joint_geometry": updated, "metadata": {"joint_protocol": {
            "kind": "fixed_count_rgb_ir_D_v1", "gaussian_count": 2,
            "trained_geometry_sha256": geometry_digest(updated)}}}
        install_joint_geometry(pc, payload)
        torch.testing.assert_close(pc.get_xyz, updated["_xyz"])
        payload["joint_geometry"]["_opacity"] += .1
        with self.assertRaisesRegex(ValueError, "checksum"):
            install_joint_geometry(pc, payload)
        torch.testing.assert_close(pc._opacity, original["_opacity"])

    @staticmethod
    def toy_gaussians():
        class Gaussian:
            def __init__(self, *args):
                self._xyz = torch.nn.Parameter(torch.tensor([[.2, .3, .9], [.4, .1, .7]]))
                self._rotation = torch.nn.Parameter(torch.ones(2, 4) * .4)
                self._scaling = torch.nn.Parameter(torch.ones(2, 3) * -.1)
                self._opacity = torch.nn.Parameter(torch.zeros(2, 1))
                self._features_dc = torch.nn.Parameter(torch.ones(2, 1, 3) * .2)
                self._features_rest = torch.nn.Parameter(torch.ones(2, 8, 3) * .01)
                self._thermal_features_dc = torch.nn.Parameter(torch.zeros(2, 1, 3))
                self._thermal_features_rest = torch.nn.Parameter(torch.zeros(2, 8, 3))
                self.active_sh_degree, self.max_sh_degree = 2, 2
            @property
            def get_xyz(self):
                return self._xyz
            @property
            def get_opacity(self):
                return self._opacity.sigmoid()
        return Gaussian()

    def test_runner_command_uses_fresh_initialization_and_D_objective(self):
        from tools.run_joint_thermal import build_joint_command
        args = SimpleNamespace(reference="old.pt", display_calibration="display.json", output="new", radiometric_dir="aligned",
            degree=2, steps=30000, start_step=3000, bound=.5, lr=.01, lr_final=.0001, regularization=.001,
            lambda_tv=0., joint_start_step=3000, lambda_rgb=1., lambda_signal=1., lambda_display=1.,
            joint_position_lr=1e-5, joint_position_lr_final=1e-7, joint_scaling_lr=1e-4,
            joint_rotation_lr=1e-4, joint_opacity_lr=1e-3, joint_rgb_lr=1e-4)
        meta = {"shared_training_protocol": {}, "geometry_iteration": 30000}
        command = build_joint_command(args, meta, {"temp_min": 250, "temp_max": 450, "temp_ref": 300}, ROOT,
                                      "scene", "geometry", "masks", "materials")
        self.assertNotIn("--stage2_checkpoint", command)
        self.assertEqual(command[command.index("--lambda_tv") + 1], "0.0")
        self.assertEqual(Path(command[1]).name, "train_thermal_joint.py")
        self.assertIn("--lambda_display", command)

    def test_actual_render_wrapper_passes_shared_geometry_gradients(self):
        from utils.sh_utils import eval_sh, SH2RGB
        class Rasterizer:
            def __init__(self, raster_settings):
                pass
            def __call__(self, **kwargs):
                output = (kwargs["colors_precomp"] * kwargs["opacities"]).mean(0)
                output = output + .01 * (kwargs["means3D"].sum() + kwargs["scales"].sum() + kwargs["rotations"][:, 0].sum())
                return output[:, None, None].expand(3, 16, 16), torch.ones(2), torch.zeros(1)
        namespace = {"torch": torch, "math": math, "GaussianModel": object,
            "GaussianRasterizationSettings": SimpleNamespace, "GaussianRasterizer": Rasterizer,
            "eval_sh": eval_sh, "SH2RGB": SH2RGB}
        render = source_function("gaussian_renderer/__init__.py", "render", namespace)
        pc = self.toy_gaussians()
        pc.get_features = torch.cat((pc._features_dc, pc._features_rest), dim=1)
        pc.get_scaling, pc.get_rotation = pc._scaling.exp(), F.normalize(pc._rotation, dim=-1)
        camera = SimpleNamespace(image_height=16, image_width=16, FoVx=1., FoVy=1.,
            world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=True)
        render(camera, pc, pipe, torch.zeros(3), 0., 0., 0., "cpu", detach_geometry=False)["render"].square().mean().backward()
        for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest"):
            self.assertGreater(float(getattr(pc, name).grad.abs().sum()), 0., name)

    def test_real_training_loop_warmup_joint_updates_saves_and_excludes_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry, scene_path = root / "geometry", root / "scene"
            geometry.mkdir(); scene_path.mkdir()
            frozen_display = display_fixture(root / "display.json")
            recorded = {"fit": ["fit"], "validation": ["validation"], "test": ["test"]}
            protocol = {"format_version": 2}
            scale = {"mode": "fit_quantile", "fit_camera_names": ["fit"], "scale": .2}
            metadata = {"geometry_sha256": "hash", "material_config_sha256": "hash", "camera_split": recorded,
                "geometry_model": str(geometry), "geometry_iteration": 30000, "source_path": str(scene_path),
                "shared_training_protocol": {"observation_domain": "raw_rjpeg", "radiometric_protocol": protocol, "loss_scale_protocol": scale}}
            reference = root / "reference" / "thermal.pt"
            reference.parent.mkdir()
            torch.save({"branch": "stage2", "config": {"temp_min": 250., "temp_max": 450., "temp_ref": 300.},
                        "metadata": metadata, "state_dict": {"poison": torch.tensor(float("nan"))}}, reference)
            initial = self.toy_gaussians()
            original = frozen_geometry_state(initial)
            cameras = [SimpleNamespace(image_name=name, camera_center=torch.tensor([.1, .1, -.5]),
                image_width=16, image_height=16, original_image=torch.ones(3, 16, 16) * .5,
                original_rgb_image=torch.ones(3, 16, 16) * .5, original_physical_image=torch.ones(3, 16, 16) * .5,
                thermal_valid_mask=torch.ones(1, 16, 16, dtype=torch.bool),
                world_view_transform=torch.eye(4), projection_matrix=torch.eye(4), full_proj_transform=torch.eye(4))
                for name in ("fit", "validation", "test")]
            class Scene:
                def __init__(self, dataset, gaussians, **kwargs):
                    self.gaussians, self.loaded_iter, self.has_rgbt, self.cameras_extent = gaussians, 30000, True, 1.
                def getTrainCameras(self):
                    return cameras[:2]
                def getTestCameras(self):
                    return cameras[2:]
            def render(camera, gaussians, pipe, background, *a, **kwargs):
                self.assertNotEqual(camera.image_name, "test")
                color = kwargs.get("override_color", gaussians._features_dc[:, 0, :] + gaussians._features_rest.mean(1))
                shape = .01 * (gaussians._xyz.mean() + gaussians._scaling.mean() + gaussians._rotation.mean())
                value = (color * gaussians.get_opacity).mean(0) + shape
                return {"render": value[:, None, None].expand(3, 16, 16), "visibility_filter": torch.ones(2, dtype=torch.bool), "radii": torch.ones(2)}
            def prepare(args, views, device):
                self.assertEqual([view.image_name for view in views], ["fit", "validation"])
                return UniformLWIRPlanckLUT(250., 450., num_temp=128)
            material = {"temperature_reference_K": 300., "document": {"confirmed": True}}
            def make_field(args, scene, device, planck, visibility, fit, config):
                for view in fit:
                    visibility(view)
                return MaterialThermalField(torch.zeros(2, dtype=torch.long), torch.ones(2, 1) * .5,
                                            torch.tensor([.95]), branch="stage2")
            base = SimpleNamespace(geometry_checkpoint_sha256=lambda *a: "hash", file_sha256=lambda *a: "hash",
                prepare_thermal_input=prepare, load_material_config=lambda *a: material, make_field=make_field,
                checkpoint_metadata=lambda args, scene, **extra: dict(metadata, **extra))
            namespace = dict(torch=torch, np=np, F=F, math=math, json=json, random=random, Path=Path, os=os,
                base=base, render=render, GaussianModel=lambda *a: initial, Scene=Scene,
                load_manifest=lambda *a: (root, {"frames": {name: {} for name in ("fit", "validation", "test")}}, "hash"),
                FrozenDisplay=lambda *a: frozen_display, TorchFrozenDisplay=TorchFrozenDisplay,
                original_rgb=lambda source, name, *a: np.ones((16, 16, 3), np.float32) * .5,
                install_loss_scale=install_loss_scale, ThermalSHResidual=ThermalSHResidual,
                held_lr=held_lr, thermal_data_loss=thermal_data_loss, masked_mean=masked_mean,
                image_loss=image_loss, joint_signal=joint_signal, sequential_joint_step=sequential_joint_step,
                frozen_geometry_state=frozen_geometry_state, geometry_digest=geometry_digest)
            for name in ("select_splits", "park_images", "render_rgb", "render_signal", "validation"):
                source_function("train_thermal_joint.py", name, namespace)
            run = source_function("train_thermal_joint.py", "run", namespace, cpu_smoke=True)
            args = SimpleNamespace(model_path=str(root / "new_D"), source_path=str(scene_path), geometry_model=str(geometry),
                geometry_iteration=30000, material_config="material.json", radiometric_dir="aligned", seed=0,
                temp_min=250., temp_max=450., temp_ref=300., loss_scale_mode="fit_quantile", steps=6,
                sh_residual_bound=.5, temperature_lr=.001, temperature_lr_final=.0001,
                environment_lr=.0001, environment_lr_final=.00001, sh_residual_lr=.01, sh_residual_lr_final=.001,
                sh_residual_start_step=3, lr_hold_fraction=.5, huber_delta=.02, environment_prior_beta=.02,
                lambda_environment=.01, lambda_sh_residual=.001, eval_every=2, save_every=4)
            options = SimpleNamespace(joint_reference=str(reference), display_calibration=str(root / "display.json"),
                joint_start_step=3, joint_position_lr=.01, joint_position_lr_final=.001,
                joint_scaling_lr=.01, joint_rotation_lr=.01, joint_opacity_lr=.01, joint_rgb_lr=.01,
                lambda_rgb=1., lambda_signal=1., lambda_display=1.)
            with patch.object(torch.cuda, "manual_seed_all"), patch.object(torch.cuda, "reset_peak_memory_stats"), \
                    patch.object(torch.cuda, "max_memory_allocated", return_value=100), \
                    patch.object(torch.cuda, "max_memory_reserved", return_value=100), redirect_stdout(io.StringIO()):
                run(args, SimpleNamespace(sh_degree=2, is_6dof=False), None, torch.device("cpu"), options)
            output = Path(args.model_path)
            warmup = torch.load(output / "thermal_joint_D_warmup.pt", map_location="cpu")
            final = torch.load(output / "thermal_joint_D_final.pt", map_location="cpu")
            selected = torch.load(output / "thermal_joint_D_selected.pt", map_location="cpu")
            for name in original:
                if name != "active_sh_degree":
                    torch.testing.assert_close(warmup["joint_geometry"][name], original[name])
            self.assertEqual(float(warmup["sh_residual"]["state_dict"]["raw"].abs().sum()), 0.)
            for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest"):
                self.assertFalse(torch.equal(final["joint_geometry"][name], original[name]), name)
            self.assertGreater(float(final["sh_residual"]["state_dict"]["raw"].abs().sum()), 0.)
            self.assertGreaterEqual(selected["step"], 3)
            self.assertFalse(selected["metadata"]["joint_protocol"]["Stage2_weights_loaded"])
            self.assertFalse(selected["metadata"]["stage2_common_endpoint"])
            install_joint_geometry(self.toy_gaussians(), selected)
            self.assertTrue((output / "training_summary.json").is_file())
            # Exercise the real evaluator restore branch, using CPU RGB rendering.
            from arguments import ModelParams, PipelineParams
            from utils.flir_radiometry import file_sha256
            from utils.loss_utils import ssim
            evaluated = []
            def evaluation_gaussian(*args):
                value = self.toy_gaussians()
                evaluated.append(value)
                return value
            def evaluation_render(camera, pc, *args, **kwargs):
                return {"render": pc._features_dc[:, 0, :].mean(0)[:, None, None].expand(3, 16, 16)}
            eval_namespace = dict(torch=torch, np=np, F=F, math=math, argparse=argparse, os=os, Path=Path,
                json=json, sys=sys, SimpleNamespace=SimpleNamespace, ModelParams=ModelParams, PipelineParams=PipelineParams,
                GaussianModel=evaluation_gaussian, Scene=Scene, install_joint_geometry=install_joint_geometry,
                geometry_checkpoint_sha256=lambda *args: "hash", file_sha256=file_sha256,
                MaterialThermalField=MaterialThermalField, render=evaluation_render,
                render_opacity=lambda *args: None, masked_mean=masked_mean, ssim=ssim,
                tqdm=lambda views, **kwargs: views)
            source_function("tools/evaluate_thermal_physics.py", "select_views", eval_namespace)
            evaluate = source_function("tools/evaluate_thermal_physics.py", "main", eval_namespace, cpu_smoke="evaluation")
            argv = ["evaluate", "--checkpoint", str(output / "thermal_joint_D_selected.pt"),
                "--modality", "rgb", "--split", "test", "--output_dir", str(output / "eval_smoke")]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                evaluate()
            torch.testing.assert_close(evaluated[0].get_xyz, selected["joint_geometry"]["_xyz"])
            report = json.loads((output / "eval_smoke/metrics.json").read_text())
            self.assertEqual(report["geometry_evaluated"], "checkpoint_joint_geometry")
            self.assertEqual(report["evaluated_stage"], "joint_D")


if __name__ == "__main__":
    unittest.main(verbosity=2)
