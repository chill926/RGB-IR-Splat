"""Run CPU protocol tests; Torch adds differentiable loss/inheritance tests."""
import importlib.util
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.thermal_training_utils import (PlateauTracker, scale_from_samples,
    held_lr, quality_status, install_loss_scale, thermal_data_loss)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


class TrainingProtocolTests(unittest.TestCase):
    def test_small_improvements_save_best_and_accumulate(self):
        tracker = PlateauTracker(absolute_delta=1e-8, relative_delta=0.002)
        self.assertEqual(tracker.update(0.00015, 0), (True, True))
        self.assertEqual(tracker.update(0.0001499, 500), (True, False))
        self.assertEqual(tracker.best_step, 500)
        self.assertEqual(tracker.stale, 1)
        self.assertEqual(tracker.update(0.0001496, 1000), (True, True))
        self.assertEqual(tracker.stale, 0)
        self.assertEqual(tracker.update(0.00016, 1500), (False, False))
        self.assertEqual(tracker.best_step, 1000)

    def test_bad_fit_plateau_is_not_quality_pass(self):
        tracker = PlateauTracker()
        for step in range(20):
            tracker.update(0.1, step)
        self.assertEqual(tracker.stale, 19)
        self.assertEqual(quality_status({"apparent_temperature_MAE_K": 0.4}, apparent_mae_K=0.1), "poor_fit")
        self.assertEqual(quality_status({}), "unassessed")

    def test_quality_requires_all_configured_metrics(self):
        metrics = {"apparent_temperature_MAE_K": 0.09, "camera_signal_RMSE": 12}
        self.assertEqual(quality_status(metrics, 0.1, 10), "poor_fit")
        self.assertEqual(quality_status(metrics, 0.1, 15), "passed")
        with self.assertRaises(ValueError):
            quality_status({}, apparent_mae_K=0.1)

    def test_fixed_signal_scale_is_affine_equivariant(self):
        import numpy as np
        samples = np.linspace(0.10, 0.14, 1001)
        scale, _, _ = scale_from_samples(samples)
        changed, _, _ = scale_from_samples(samples * 5 + 2)
        self.assertAlmostEqual(changed, scale * 5)
        self.assertAlmostEqual(0.002 / scale, 0.01 / changed)

    def test_constant_scene_floor_and_invalid_inputs(self):
        self.assertEqual(scale_from_samples([0.13] * 20)[0], 1e-4)
        for invalid in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                scale_from_samples(invalid)

    def test_hold_then_decay_reaches_requested_end(self):
        self.assertAlmostEqual(held_lr(1, 30000, 1e-3, 1e-5, 0.5), 1e-3)
        self.assertAlmostEqual(held_lr(15000, 30000, 1e-3, 1e-5, 0.5), 1e-3)
        self.assertAlmostEqual(held_lr(22500, 30000, 1e-3, 1e-5, 0.5), 1e-4)
        self.assertAlmostEqual(held_lr(30000, 30000, 1e-3, 1e-5, 0.5), 1e-5)

    def test_invalid_tracker_metrics_rejected(self):
        for value in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                PlateauTracker().update(value, 0)

    def test_branch_inherits_loss_units_from_old_and_new_checkpoints(self):
        # Isolate the actual loader from CUDA imports, exercising its checkpoint contract.
        path = Path(__file__).resolve().parents[1] / "train_thermal_physics.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "apply_stage2_protocol")
        for shared, expected in (({"observation_domain": "normalized_dn"},
                                 ("legacy", 1.0, 0.0, None)),
                                ({"observation_domain": "raw_rjpeg", "radiometric_protocol": {},
                                  "loss_scale_mode": "fit_quantile", "tv_temperature_scale_K": 10.0,
                                  "lr_hold_fraction": 0.5, "loss_scale_protocol": {"scale": 0.02}},
                                 ("fit_quantile", 10.0, 0.5, {"scale": 0.02}))):
            checkpoint = {"metadata": {"shared_training_protocol": shared}}
            namespace = {"torch": SimpleNamespace(load=lambda *a, **kw: checkpoint)}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
            args = SimpleNamespace(stage="branch", stage2_checkpoint="reference.pt", radiometric_dir="",
                environment_lr=1e-4, loss_scale_mode="auto", tv_temperature_scale_K=10.0, lr_hold_fraction=0.5)
            namespace["apply_stage2_protocol"](args)
            self.assertEqual((args.loss_scale_mode, args.tv_temperature_scale_K,
                              args.lr_hold_fraction, args.loss_scale_protocol), expected)

    @unittest.skipUnless(TORCH_AVAILABLE, "Torch unavailable; run this test on the physir server")
    def test_loss_gradient_and_legacy_checkpoint_units(self):
        p = torch.tensor([0.1301, 0.131], requires_grad=True)
        t = torch.tensor([0.13, 0.13])
        legacy = thermal_data_loss(p, t, {})
        expected = torch.nn.functional.smooth_l1_loss(p, t, beta=0.02)
        self.assertTrue(torch.allclose(legacy, expected))
        loss = thermal_data_loss(p, t, {"huber_delta": 0.02, "loss_scale_protocol": {"scale": 0.02}})
        grad = torch.autograd.grad(loss, p)[0]
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertTrue(bool((grad > 0).all()))
        equivalent = thermal_data_loss(p * 5 + 2, t * 5 + 2,
            {"huber_delta": 0.02, "loss_scale_protocol": {"scale": 0.1}})
        self.assertAlmostEqual(float(equivalent), float(loss), places=5)

    @unittest.skipUnless(TORCH_AVAILABLE, "Torch unavailable; run this test on the physir server")
    def test_fit_only_scale_and_checkpoint_reuse(self):
        camera = SimpleNamespace(image_name="fit", original_physical_image=None,
            original_image=torch.linspace(0.10, 0.14, 32 * 32).reshape(1, 32, 32))
        args = SimpleNamespace(loss_scale_mode="auto", observation_domain="raw_rjpeg", loss_scale_floor=1e-4)
        install_loss_scale(args, [camera])
        recorded = args.loss_scale_protocol.copy()
        camera.original_image.fill_(1e6)
        install_loss_scale(args, [camera])
        self.assertEqual(recorded, args.loss_scale_protocol)
        self.assertFalse(recorded["held_out_pixels_used"])
        camera.image_name = "validation"
        with self.assertRaises(ValueError):
            install_loss_scale(args, [camera])


if __name__ == "__main__":
    unittest.main(verbosity=2)
