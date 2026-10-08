"""CPU gradient, persistence, and rasterizer-interface tests for G-alpha.

The rasterizer test uses a differentiable CPU stand-in; real CUDA rasterization
still needs a short server smoke run. --require_torch prevents skipped preflight.
"""
import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if "--require_torch" in sys.argv:
    sys.argv.remove("--require_torch")
    if not TORCH_AVAILABLE:
        raise SystemExit("G-alpha preflight requires Torch; use the physir environment")
if TORCH_AVAILABLE:
    import torch
    from utils.thermal_opacity import IROpacityCorrection, render_opacity


def source_function(relative, name, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class OpacityMathTests(unittest.TestCase):
    def test_logit_cap_implies_small_absolute_opacity_change(self):
        base = np.linspace(0, 1, 10001)
        clipped = np.clip(base, 1e-6, 1 - 1e-6)
        logits = np.log(clipped) - np.log1p(-clipped)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        for shift in (-0.2, 0, 0.2):
            corrected = np.clip(base + (sigmoid(logits + shift) - sigmoid(logits)), 0, 1)
            self.assertLessEqual(np.max(np.abs(corrected - base)), math.tanh(0.2 / 4) + 1e-12)
            if shift == 0:
                np.testing.assert_array_equal(corrected, base)

    def test_recolor_aggregates_signal_RMSE_from_squared_errors(self):
        summarize = source_function("tools/recolor_thermal_renders.py", "summarize", {"np": np, "math": math})
        rows = [{"image_name": "a", "MSE": 1., "signal_MSE": 1.,
                 "mapping_roundtrip_MSE": 0.04, "camera_signal_RMSE": 2.},
                {"image_name": "b", "MSE": 9., "signal_MSE": 9.,
                 "mapping_roundtrip_MSE": 0.16, "camera_signal_RMSE": 6.}]
        metrics = summarize(rows)
        self.assertAlmostEqual(metrics["camera_signal_RMSE"], math.sqrt(20))
        self.assertAlmostEqual(metrics["signal_RMSE"], math.sqrt(5))

    def test_galpha_can_change_TV_LR_while_C_R_K_inherit(self):
        shared = {"observation_domain": "raw_rjpeg", "radiometric_protocol": {},
                  "loss_scale_mode": "fit_quantile", "loss_scale_protocol": {"scale": 0.02},
                  "lambda_tv": 0.0, "temperature_lr": 0.001, "temperature_lr_final": 1e-5}
        namespace = {"torch": SimpleNamespace(load=lambda *a, **k: {"metadata": {"shared_training_protocol": shared}})}
        loader = source_function("train_thermal_physics.py", "apply_stage2_protocol", namespace)
        for stage in ("galpha", "branch"):
            args = SimpleNamespace(stage=stage, stage2_checkpoint="reference", radiometric_dir="",
                environment_lr=1e-4, lambda_tv=0.001, temperature_lr=1e-4, temperature_lr_final=1e-6)
            loader(args)
            expected = (0.001, 1e-4, 1e-6) if stage == "galpha" else (0., 0.001, 1e-5)
            self.assertEqual((args.lambda_tv, args.temperature_lr, args.temperature_lr_final), expected)
            self.assertEqual(args.loss_scale_protocol, {"scale": 0.02})


@unittest.skipUnless(TORCH_AVAILABLE, "Torch unavailable")
class OpacityTorchTests(unittest.TestCase):
    def setUp(self):
        self.base = torch.tensor([[0.01], [0.4], [0.7], [0.999]])
        self.support = torch.tensor([False, True, True, False])

    def test_initial_identity_bounds_and_unseen_opacity(self):
        field = IROpacityCorrection(self.base, self.support, trainable=True)
        self.assertTrue(torch.equal(field.opacity, self.base))
        with torch.no_grad():
            field.raw.copy_(torch.tensor([[10.], [-10.], [10.], [-10.]]))
        self.assertTrue(torch.equal(field.opacity[~self.support], self.base[~self.support]))
        self.assertLessEqual(float((field.opacity - self.base).abs().max()), math.tanh(0.05) + 2e-7)
        # Compare in the model dtype: float32(0.2) is slightly larger than
        # Python's float64 0.2, which otherwise falsely fails at saturation.
        delta = field.logit_delta
        bound = delta.new_tensor(field.logit_bound)
        self.assertTrue(bool((delta.abs() <= bound).all()))

    def test_fit_gradients_and_reference_opacity_frozen(self):
        base = self.base.clone().requires_grad_()
        field = IROpacityCorrection(base, self.support, trainable=True)
        (field.opacity.square().sum() + 0.01 * field.regularizer()).backward()
        self.assertIsNone(base.grad)
        self.assertTrue(bool((field.raw.grad[self.support].abs() > 0).all()))
        self.assertTrue(bool((field.raw.grad[~self.support] == 0).all()))

    def test_optimizer_can_fit_opacity_within_budget(self):
        field = IROpacityCorrection(self.base, self.support, trainable=True)
        target = self.base + torch.tensor([[0.], [0.02], [-0.02], [0.]])
        optimizer = torch.optim.Adam([field.raw], lr=0.05)
        initial = float((field.opacity - target).square().mean())
        for _ in range(100):
            optimizer.zero_grad()
            loss = (field.opacity - target).square().mean() + 1e-6 * field.regularizer()
            loss.backward()
            optimizer.step()
        self.assertLess(float((field.opacity - target).square().mean()), initial * 0.01)
        self.assertTrue(torch.equal(field.base_opacity, self.base))

    def test_checkpoint_roundtrip_and_C_R_K_freeze(self):
        field = IROpacityCorrection(self.base, self.support, trainable=True)
        with torch.no_grad():
            field.raw[self.support] = 0.5
        saver = source_function("utils/thermal_physics.py", "save_thermal_checkpoint", {"os": os, "torch": torch})
        model = SimpleNamespace(export_state=lambda: {"branch": "stage2", "state_dict": {}})
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "best.pt")
            saver(path, model, 500, {"example": True}, {"_opacity": self.base}, field)
            checkpoint = torch.load(path, map_location="cpu")
        frozen = IROpacityCorrection.from_checkpoint(checkpoint, self.base, trainable=False)
        self.assertFalse(frozen.raw.requires_grad)
        self.assertTrue(torch.equal(frozen.opacity, field.opacity))
        self.assertEqual(checkpoint["step"], 500)
        self.assertTrue(torch.equal(checkpoint["frozen_geometry"]["_opacity"], self.base))
        with self.assertRaisesRegex(ValueError, "differs"):
            IROpacityCorrection.from_checkpoint(checkpoint, self.base + 0.001)
        self.assertIsNone(IROpacityCorrection.from_checkpoint({}, self.base))

    def test_invalid_states_fail(self):
        for bound in (0, -0.2, float("nan"), 3):
            with self.assertRaises(ValueError):
                IROpacityCorrection(self.base, self.support, bound)
        with self.assertRaises(ValueError):
            IROpacityCorrection(self.base, [True])
        with self.assertRaises(ValueError):
            IROpacityCorrection(self.base, torch.zeros(4, dtype=torch.bool), trainable=True)

    def test_renderer_override_has_gradients_with_detached_RGB_geometry(self):
        class Rasterizer:
            def __init__(self, raster_settings):
                self.settings = raster_settings

            def __call__(self, **kwargs):
                color = (kwargs["colors_precomp"] * kwargs["opacities"]).sum(dim=0)
                # Depend on positions as well, to verify geometry detachment.
                color = color + 0.01 * kwargs["means3D"].sum()
                rgb = color[:, None, None].expand(3, self.settings.image_height, self.settings.image_width)
                return rgb, torch.ones(len(kwargs["means3D"])), torch.zeros(1)

        namespace = {"torch": torch, "math": math, "GaussianModel": object,
            "GaussianRasterizationSettings": SimpleNamespace, "GaussianRasterizer": Rasterizer}
        render = source_function("gaussian_renderer/__init__.py", "render", namespace)
        xyz = torch.nn.Parameter(torch.ones(4, 3))
        rgb_opacity = torch.nn.Parameter(self.base.clone())
        pc = SimpleNamespace(get_xyz=xyz, get_opacity=rgb_opacity,
            get_scaling=torch.ones(4, 3), get_rotation=torch.ones(4, 4),
            get_features=torch.zeros(4, 1, 3), _features_dc=torch.zeros(4, 1, 3), active_sh_degree=0)
        camera = SimpleNamespace(image_height=2, image_width=3, FoVx=1., FoVy=1.,
            world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        field = IROpacityCorrection(rgb_opacity, self.support, trainable=True)
        colors = torch.full((4, 3), 0.13)
        baseline = render(camera, pc, pipe, torch.zeros(3), 0., 0., 0., "cpu",
            override_color=colors, detach_geometry=True)["render"]
        output = render(camera, pc, pipe, torch.zeros(3), 0., 0., 0., "cpu",
            override_color=colors, detach_geometry=True, override_opacity=render_opacity(field))["render"]
        self.assertTrue(torch.equal(output, baseline))
        output.sum().backward()
        self.assertTrue(bool((field.raw.grad[self.support].abs() > 0).all()))
        self.assertIsNone(rgb_opacity.grad)
        self.assertIsNone(xyz.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
