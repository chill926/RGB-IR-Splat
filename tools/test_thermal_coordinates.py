"""CPU regressions for weighted initialization and frozen one-pass coordinates."""
import json
import ast
import importlib.util
import io
from pathlib import Path
import sys
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
from utils.thermal_coordinates import (sample_native_signal, validate_profile, profile_sha256,
    fit_alignment, validate_manifest_coordinates, load_valid_mask)
from utils.flir_radiometry import file_sha256, normalize_signal, resize_signal
from tools.prepare_rgbt_radiometry import prepare

CALIBRATIONS = [dict(PlanckR1=12487.9970703125, PlanckR2=0.02452719397842884,
    PlanckB=1345.5999755859375, PlanckF=1.600000023841858, PlanckO=-6726)]


def rjpeg_bytes(raw, calibration, swapped=False):
    """Small synthetic FLIR fixture; self-contained for server preflight."""
    height, width = raw.shape
    header = bytearray(32); struct.pack_into("<HHH", header, 0, 2, width, height)
    payload = io.BytesIO(); Image.fromarray(raw.byteswap() if swapped else raw).save(payload, format="PNG")
    raw_record = bytes(header) + payload.getvalue()
    camera = bytearray(0x340); struct.pack_into("<H", camera, 0, 2)
    for name, offset in dict(PlanckR1=0x58, PlanckB=0x5C, PlanckF=0x60, PlanckR2=0x30C).items():
        struct.pack_into("<f", camera, offset, calibration[name])
    struct.pack_into("<i", camera, 0x308, calibration["PlanckO"])
    struct.pack_into("<H", camera, 0x338, int(np.median(raw)))
    fff = bytearray(128); fff[:4] = b"FFF\x00"; struct.pack_into("<III", fff, 0x14, 100, 64, 2)
    struct.pack_into("<H", fff, 64, 1); struct.pack_into("<II", fff, 76, 128, len(raw_record))
    struct.pack_into("<H", fff, 96, 0x20); struct.pack_into("<II", fff, 108, 128 + len(raw_record), len(camera))
    data = b"FLIR\x00\x01\x00\x00" + bytes(fff) + raw_record + bytes(camera)
    return b"\xff\xd8\xff\xe1" + struct.pack(">H", len(data) + 2) + data + b"\xff\xd9"

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if "--require_torch" in sys.argv:
    sys.argv.remove("--require_torch")
    if not TORCH_AVAILABLE:
        raise SystemExit("Coordinate preflight requires Torch in the training environment")
if TORCH_AVAILABLE:
    import torch
    from utils.thermal_physics import map_thermal_observations_to_gaussians
    from utils.thermal_observations import prepare_thermal_input, radiometric_image_errors
    from utils.thermal_training_utils import thermal_data_loss, masked_mean, install_loss_scale


def profile(size=(20, 16), offset=(-1.5, .75)):
    split = {"fit": ["f0", "f1", "f2"], "validation": ["v"], "test": ["t"]}
    return {"format_version": 1, "kind": "fit_only_signal_translation",
        "reference_image_size": list(size), "offset_reference_pixels": list(offset),
        "sampling_convention": "half_pixel_centres_native_to_target", "fit_policy": "fit_pixels_only",
        "camera_split": split, "fit_camera_names": split["fit"],
        "fit_references": {name: {"raw_sha256": "x", "thermal_reference_sha256": "x"} for name in split["fit"]},
        "held_out_pixels_used_for_fitting": False}


class CoordinateTests(unittest.TestCase):
    def test_zero_offset_matches_half_pixel_bilinear_resize(self):
        yy, xx = np.mgrid[:8, :10]
        source = (10 + xx + 2 * yy).astype(np.float32)
        actual, valid = sample_native_signal(source, (20, 16), profile(offset=(0, 0)))
        np.testing.assert_allclose(actual, resize_signal(source, (20, 16)), atol=1e-6)
        self.assertTrue(valid.all())

    def test_shift_sign_and_single_pass_coordinates_on_linear_ramp(self):
        yy, xx = np.mgrid[:8, :10]
        source = (10 + 2 * xx + 3 * yy).astype(float)
        p = profile(offset=(-1.5, .75))
        actual, valid = sample_native_signal(source, (20, 16), p)
        tx = (np.arange(20) + .5) * .5 - .5 - .75
        ty = (np.arange(16) + .5) * .5 - .5 + .375
        expected = 10 + 2 * np.clip(tx, 0, 9)[None] + 3 * np.clip(ty, 0, 7)[:, None]
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertFalse(valid[:, 0].any())
        self.assertFalse(valid[-1].any())

    def test_profile_rejects_leakage_and_invalid_offsets(self):
        p = profile(); p["camera_split"]["test"].append("f0")
        with self.assertRaises(ValueError): validate_profile(p)
        p = profile(); p["held_out_pixels_used_for_fitting"] = True
        with self.assertRaises(ValueError): validate_profile(p)
        p = profile(); p["offset_reference_pixels"][0] = float("nan")
        with self.assertRaises(ValueError): validate_profile(p)

    def test_fit_recovers_translation_without_reading_held_out_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = Path(temp)
            for folder in ("raw_images", "thermal/train", "thermal/test"):
                (scene / folder).mkdir(parents=True)
            truth = profile(size=(128, 128), offset=(-1.75, .75))
            rng = np.random.RandomState(21)
            signals = {}
            yy, xx = np.mgrid[:64, :64]
            for index, name in enumerate(truth["fit_camera_names"]):
                field = .25 + .001 * xx + .0015 * yy
                for _ in range(8):
                    x, y = rng.uniform(8, 56, 2)
                    field += rng.uniform(.1, .25) * np.exp(-((xx-x)**2 + (yy-y)**2) / rng.uniform(12, 45))
                field = np.clip(field, .1, .9) * 255
                signals[name] = field
                image, _ = sample_native_signal(field, (128, 128), truth)
                rgb = np.repeat(np.rint(image).astype(np.uint8)[..., None], 3, -1)
                Image.fromarray(rgb).save(scene / "thermal/train" / (name + ".png"))
                (scene / "raw_images" / (name + ".jpg")).write_bytes(b"fit raw hash input")
            for name, role in (("v", "train"), ("t", "test")):
                (scene / "thermal" / role / (name + ".png")).write_bytes(b"DO NOT READ HELD OUT PIXELS")
                (scene / "raw_images" / (name + ".jpg")).write_bytes(b"DO NOT DECODE HELD OUT SIGNAL")
            calls = []
            def decode(path, *a):
                calls.append(Path(path).stem)
                return signals[Path(path).stem], {"PlanckO": 0}, {}
            palette = np.c_[np.arange(256), np.full(256, 128), np.full(256, 128)].tolist()
            with patch("utils.thermal_coordinates.decode_flir_rjpeg", side_effect=decode), \
                    patch("utils.thermal_coordinates.read_flir_display_metadata", return_value={"palette_ycrcb": palette}):
                learned = fit_alignment(scene, truth["camera_split"], stride=2, max_shift=5)
            self.assertEqual(calls, truth["fit_camera_names"])
            np.testing.assert_allclose(learned["offset_reference_pixels"], [-1.75, .75], atol=.3)
            self.assertFalse(learned["held_out_pixels_used_for_fitting"])


@unittest.skipUnless(TORCH_AVAILABLE, "Torch unavailable")
class TorchCoordinateTests(unittest.TestCase):
    def test_weighted_initialization_is_invariant_to_support_scale(self):
        camera = SimpleNamespace(original_physical_image=torch.full((1, 2, 2), .13), original_image=None)
        for opacity in (.02, 1e-9, .8):
            gaussians = SimpleNamespace(get_xyz=torch.zeros(1, 3), get_opacity=torch.tensor([[opacity]]))
            with patch("utils.thermal_physics._project_gaussians", side_effect=lambda *a, **k: (torch.zeros(1, 3), torch.tensor([True]))):
                output = map_thermal_observations_to_gaussians(gaussians, [camera],
                    visibility_provider=lambda c: (torch.tensor([True]), torch.tensor([3.])), fallback_radiance=.17)
            self.assertAlmostEqual(float(output[0, 0]), .13, places=6)

    def test_zero_or_invalid_support_uses_fallback(self):
        for opacity, mask in ((0., None), (.02, torch.zeros(1, 2, 2, dtype=torch.bool))):
            camera = SimpleNamespace(original_physical_image=torch.full((1, 2, 2), .13), original_image=None,
                                     thermal_valid_mask=mask)
            gaussians = SimpleNamespace(get_xyz=torch.zeros(1, 3), get_opacity=torch.tensor([[opacity]]))
            with patch("utils.thermal_physics._project_gaussians", side_effect=lambda *a, **k: (torch.zeros(1, 3), torch.tensor([True]))):
                output = map_thermal_observations_to_gaussians(gaussians, [camera], fallback_radiance=.17)
            self.assertAlmostEqual(float(output[0, 0]), .17, places=6)

    def test_masked_loss_excludes_border_gradients_and_statistics(self):
        prediction = torch.tensor([[[.9, .2, .3]]], requires_grad=True)
        target = torch.tensor([[[0., .2, .2]]])
        mask = torch.tensor([[[False, True, True]]])
        actual = thermal_data_loss(prediction, target, {}, mask)
        expected = torch.nn.functional.smooth_l1_loss(prediction[..., 1:], target[..., 1:], beta=.02)
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        self.assertEqual(float(prediction.grad[..., 0]), 0)
        self.assertGreater(float(prediction.grad[..., 2]), 0)
        self.assertAlmostEqual(float(masked_mean((prediction-target).abs(), mask).detach()), .05, places=6)
        with self.assertRaises(ValueError): masked_mean(prediction, torch.zeros_like(mask))

    def test_training_validation_uses_the_same_valid_mask(self):
        import math
        from utils.thermal_sh_residual import thermal_view_radiance
        image = torch.tensor([[[.9, .2, .2]]]).repeat(3, 1, 1)
        namespace = {"torch": torch, "math": math, "F": torch.nn.functional,
            "render": lambda *a, **k: {"render": image}, "thermal_view_radiance": thermal_view_radiance,
            "render_opacity": lambda x: None, "thermal_data_loss": thermal_data_loss,
            "masked_mean": masked_mean, "observation_to_model_radiance": lambda x, *a: x}
        path = ROOT / "train_thermal_physics.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validation_metrics")
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        validate = namespace["validation_metrics"]
        camera = SimpleNamespace(camera_center=torch.zeros(3), original_physical_image=torch.full((1, 1, 3), .2),
            original_image=None, thermal_valid_mask=torch.tensor([[[False, True, True]]]))
        field = SimpleNamespace(radiance=lambda x: torch.full((1, 1), .2))
        args = SimpleNamespace(observation_domain="normalized_dn", huber_delta=.02)
        result = validate(field, None, [camera], SimpleNamespace(get_xyz=torch.zeros(1, 3)), None,
            torch.zeros(3), SimpleNamespace(is_6dof=False), args)
        self.assertEqual(result["signal_MSE"], 0)
        self.assertEqual(result["signal_MAE"], 0)
        self.assertEqual(result["radiance_loss"], 0)

    def make_aligned_scene(self, root):
        scene = root / "Scene"
        (scene / "raw_images").mkdir(parents=True)
        for role in ("train", "test"): (scene / "thermal" / role).mkdir(parents=True)
        p = profile()
        rng = np.random.RandomState(3)
        sources = {}
        for name in sum(p["camera_split"].values(), []):
            role = "test" if name == "t" else "train"
            raw = (12000 + rng.randint(0, 100, (8, 10))).astype(np.uint16)
            sources[name] = raw.astype(float) + CALIBRATIONS[0]["PlanckO"]
            rp = scene / "raw_images" / (name + ".jpg")
            rp.write_bytes(rjpeg_bytes(raw, CALIBRATIONS[0], swapped=True))
            tp = scene / "thermal" / role / (name + ".png")
            Image.fromarray(np.full((16, 20, 3), 100, np.uint8)).save(tp)
            if name in p["fit_references"]:
                p["fit_references"][name] = {"raw_sha256": file_sha256(rp), "thermal_reference_sha256": file_sha256(tp)}
        directory = root / "aligned"
        manifest = prepare(scene, directory, alignment=p)
        return scene, directory, manifest, p, sources

    def test_manifest_mask_hash_and_profile_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            scene, directory, manifest, p, sources = self.make_aligned_scene(Path(temp))
            self.assertEqual(manifest["format_version"], 2)
            self.assertEqual(validate_manifest_coordinates(manifest), p)
            row = manifest["frames"]["f0"]
            self.assertLess(float(load_valid_mask(directory, row).mean()), 1)
            np.save(directory / row["valid_mask_file"], np.zeros((16, 20), dtype=bool))
            with self.assertRaisesRegex(ValueError, "changed"): load_valid_mask(directory, row)
            manifest["coordinate_alignment"]["offset_reference_pixels"][0] += 1
            with self.assertRaisesRegex(ValueError, "hash"): validate_manifest_coordinates(manifest)

    def test_loader_resamples_native_once_at_requested_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            scene, directory, manifest, p, sources = self.make_aligned_scene(Path(temp))
            camera = SimpleNamespace(image_name="f0", image_width=10, image_height=8)
            args = SimpleNamespace(observation_domain="raw_rjpeg", radiometric_dir=str(directory),
                source_path=str(scene), temp_min=250., temp_max=450., fit_camera_names=p["fit_camera_names"])
            prepare_thermal_input(args, [camera], "cpu")
            signal, valid = sample_native_signal(sources["f0"], (10, 8), p)
            expected = normalize_signal(signal, manifest["signal_calibration"], 250, 450)
            np.testing.assert_array_equal(camera.original_physical_image[0].numpy(), expected)
            np.testing.assert_array_equal(camera.thermal_valid_mask[0].numpy(), valid)
            twice = resize_signal(np.load(directory / manifest["frames"]["f0"]["signal_file"]), (10, 8))
            self.assertGreater(float(np.max(np.abs(signal - twice))), .1)
            args.fit_camera_names = ["v"]
            with self.assertRaisesRegex(ValueError, "fit split"): prepare_thermal_input(args, [camera], "cpu")

    def test_old_protocol_cannot_silently_use_new_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            scene, directory, manifest, p, sources = self.make_aligned_scene(Path(temp))
            old = Path(temp) / "legacy"
            prepare(scene, old)
            old_args = SimpleNamespace(observation_domain="raw_rjpeg", radiometric_dir=str(old),
                source_path=str(scene), temp_min=250., temp_max=450.)
            camera = SimpleNamespace(image_name="f0", image_width=20, image_height=16)
            prepare_thermal_input(old_args, [camera], "cpu")
            self.assertIsNone(camera.thermal_valid_mask)
            old_args.radiometric_dir = str(directory)
            with self.assertRaisesRegex(ValueError, "differs"): prepare_thermal_input(old_args, [camera], "cpu")

    def test_loss_scale_uses_fit_valid_pixels_only(self):
        values = torch.linspace(.12, .15, 32 * 32).reshape(1, 32, 32)
        values[0, 0, 0] = 1000
        mask = torch.ones(1, 32, 32, dtype=torch.bool); mask[0, 0, 0] = False
        camera = SimpleNamespace(image_name="fit", original_physical_image=values, thermal_valid_mask=mask)
        args = SimpleNamespace(loss_scale_mode="auto", observation_domain="raw_rjpeg", loss_scale_floor=1e-4)
        install_loss_scale(args, [camera])
        self.assertLess(args.loss_scale_protocol["scale"], .04)
        self.assertEqual(args.loss_scale_protocol["pixel_policy"], "valid_native_FOV_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
