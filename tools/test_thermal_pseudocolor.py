"""CPU tests for frozen pseudo-color recovery and original-image evaluation."""
import ast
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from tools.test_thermal_radiometry import rjpeg_bytes, CALIBRATIONS
from tools.prepare_rgbt_radiometry import prepare
from tools.recolor_thermal_renders import recolor
from utils.flir_radiometry import _fff_from_jpeg, file_sha256, normalize_signal, read_flir_display_metadata
from utils.thermal_pseudocolor import (FrozenDisplay, calibrate_display, colorize_signal,
    fit_response, isotonic_response, normalized_to_signal, original_rgb, palette_rgb,
    rgb_metrics)


def capture_bytes(raw, palette, median, span):
    fff = _fff_from_jpeg(rjpeg_bytes(raw, CALIBRATIONS[0]))
    header = bytearray(fff[:64]); struct.pack_into("<I", header, 0x1C, 3)
    entries = []
    payload = bytearray(fff[128:])
    for index in range(2):
        entry = bytearray(fff[64 + index * 32:96 + index * 32])
        start, length = struct.unpack_from("<II", entry, 12)
        if struct.unpack_from("<H", entry)[0] == 0x20:
            struct.pack_into("<H", payload, start - 128 + 0x338, median)
            struct.pack_into("<H", payload, start - 128 + 0x33C, span)
        struct.pack_into("<II", entry, 12, start + 32, length)
        entries.append(bytes(entry))
    record = bytearray(112); struct.pack_into("<I", record, 0, len(palette))
    record[26], record[27] = 0, 2
    record[80:84] = b"Test"
    record = bytes(record) + palette.tobytes()
    entry = bytearray(32); struct.pack_into("<H", entry, 0, 0x22)
    struct.pack_into("<II", entry, 12, len(fff) + 32, len(record))
    new = bytes(header) + b"".join(entries) + bytes(entry) + bytes(payload) + record
    app = b"FLIR\x00\x01\x00\x00" + new
    return b"\xff\xd8\xff\xe1" + struct.pack(">H", len(app) + 2) + app + b"\xff\xd9"


class PseudocolorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, unseen_palette=False):
        scene = self.root / "AnyScene"
        (scene / "raw_images").mkdir(parents=True)
        palette = np.stack((np.linspace(16, 235, 224), np.full(224, 128), np.full(224, 128)), 1).round().astype(np.uint8)
        z = np.linspace(.1, .9, 64 * 96).reshape(64, 96)
        for index, (name, split) in enumerate((("fit", "train"), ("hold", "train"), ("test", "test"))):
            (scene / "thermal" / split).mkdir(parents=True, exist_ok=True)
            median, span = 12000 + index * 100, 800
            raw = np.round(median + (z - .5) * span).astype(np.uint16)
            colors = palette.copy()
            if unseen_palette and name == "test":
                colors[:, 1] = 160
            rgb_table = palette_rgb(colors, "limited")
            actual_z = (raw.astype(float) - median) / span + .5
            rgb = np.stack([np.interp(actual_z ** 2, np.linspace(0, 1, len(colors)), rgb_table[:, channel])
                            for channel in range(3)], -1)
            Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(scene / "thermal" / split / f"{name}.png")
            (scene / "raw_images" / f"{name}.jpg").write_bytes(capture_bytes(raw, colors, median, span))
        directory = scene / "radiometric"
        manifest = prepare(scene, directory, byte_order="native")
        split = {"fit": ["fit"], "validation": ["hold"], "test": ["test"]}
        protocol = {"manifest_sha256": file_sha256(directory / "manifest.json"),
                    "signal_calibration": {key: float(value) for key, value in manifest["signal_calibration"].items()},
                    "normalization_temperature_bounds_K": [250., 450.]}
        return scene, directory, manifest, split, protocol

    def test_weighted_monotonic_fit(self):
        np.testing.assert_allclose(isotonic_response([0, .8, .2, 1], [1, 1, 3, 1]), [0, .35, .35, 1])
        with self.assertRaises(ValueError):
            isotonic_response([0, 1], [1, -1])

    def test_palette_video_range_and_metadata(self):
        table = palette_rgb([[16, 128, 128], [235, 128, 128]], "limited")
        np.testing.assert_allclose(table, [[0, 0, 0], [1, 1, 1]], atol=1e-7)
        scene, directory, manifest, split, protocol = self.fixture()
        display = read_flir_display_metadata(scene / "raw_images/hold.jpg")
        self.assertEqual(display["raw_value_range"], 800)
        self.assertEqual(display["signal_window_center"], 12100 - 6726)
        self.assertEqual(len(display["palette_ycrcb"]), 224)

    def test_fit_never_reads_held_out_pixels(self):
        scene, directory, manifest, split, protocol = self.fixture()
        import utils.thermal_pseudocolor as module
        original = module.original_rgb
        called = []
        def fit_only(source, name, record, shape=None):
            called.append(name)
            self.assertEqual(name, "fit")
            return original(source, name, record, shape)
        with mock.patch.object(module, "original_rgb", side_effect=fit_only):
            document = calibrate_display(scene, directory, split["fit"], self.root / "display.json",
                                         split, protocol, bins=128, stride=1)
        self.assertEqual(called, ["fit"])
        self.assertEqual(document["fit_camera_names"], ["fit"])

    def test_held_out_fit_names_rejected(self):
        scene, directory, manifest, split, protocol = self.fixture()
        with self.assertRaisesRegex(ValueError, "exactly"):
            calibrate_display(scene, directory, ["hold"], self.root / "bad.json", split, protocol)
        with self.assertRaisesRegex(ValueError, "training"):
            calibrate_display(scene, directory, ["test"], self.root / "bad.json")

    def test_fixed_mapping_reproduces_held_out_and_has_no_image_stretch(self):
        scene, directory, manifest, split, protocol = self.fixture()
        path = self.root / "display.json"
        calibrate_display(scene, directory, split["fit"], path, split, protocol, bins=128, stride=1)
        display = FrozenDisplay(path, protocol, split["fit"])
        q = np.load(directory / manifest["frames"]["hold"]["signal_file"])
        y = normalize_signal(q, protocol["signal_calibration"], 250, 450)
        restored, _ = display.colorize(y, "hold", [250, 450])
        target = original_rgb(scene, "hold", manifest["frames"]["hold"])
        self.assertGreater(rgb_metrics(restored, target)["PSNR"], 35)
        pixel = y[20, 30]
        a = np.full((4, 4), pixel); b = np.zeros((4, 4)); b[0, 0] = pixel
        ra, _ = display.colorize(a, "hold", [250, 450]); rb, _ = display.colorize(b, "hold", [250, 450])
        np.testing.assert_array_equal(ra[0, 0], rb[0, 0])
        arbitrary, _ = display.colorize_with_settings(a, [250, 450], display.document["frame_display_metadata"]["hold"])
        np.testing.assert_array_equal(arbitrary, ra)

    def test_unseen_metadata_palette_uses_training_response(self):
        scene, directory, manifest, split, protocol = self.fixture(unseen_palette=True)
        path = self.root / "display.json"
        document = calibrate_display(scene, directory, split["fit"], path, split, protocol, bins=128, stride=1)
        key = document["frame_display_metadata"]["test"]["palette_sha256"]
        self.assertEqual(document["groups"][key]["fit_names"], [])
        self.assertEqual(document["groups"][key]["calibration_support"], "metadata_palette_with_shared_fit_response")
        self.assertNotEqual(key, document["groups"][key]["response_source_palette"])

    def test_calibration_provenance_and_duplicate_output(self):
        scene, directory, manifest, split, protocol = self.fixture()
        path = self.root / "display.json"
        calibrate_display(scene, directory, split["fit"], path, split, protocol)
        with self.assertRaises(FileExistsError):
            calibrate_display(scene, directory, split["fit"], path, split, protocol)
        with self.assertRaisesRegex(ValueError, "fit cameras"):
            FrozenDisplay(path, protocol, ["hold"])
        bad = dict(protocol, manifest_sha256="incorrect")
        with self.assertRaisesRegex(ValueError, "provenance"):
            FrozenDisplay(path, bad)

    def test_signal_denormalization_and_rejection(self):
        q = np.array([[5000., 5500.]])
        y = normalize_signal(q, CALIBRATIONS[0], 250, 450)
        np.testing.assert_allclose(normalized_to_signal(y[None], CALIBRATIONS[0], [250, 450]), q, atol=.001)
        with self.assertRaises(ValueError):
            normalized_to_signal(np.array([[2.]]), CALIBRATIONS[0], [250, 450])

    def test_monotonic_curve_tails_and_range(self):
        z = np.linspace(.2, .8, 1000)
        response = fit_response(z, z ** 2, bins=64)
        display = {"signal_window_center": 0., "raw_value_range": 1.}
        group = {"response": response, "palette_rgb": [[0, 0, 0], [1, 1, 1]]}
        rgb, fraction = colorize_signal(np.array([[-100., 0., 100.]]), display, group)
        self.assertGreater(fraction, 0)
        self.assertTrue(np.all(np.diff(rgb[0, :, 0]) >= 0))
        self.assertEqual(rgb[0, 0, 0], 0)
        self.assertEqual(rgb[0, -1, 0], 1)

    def test_cpu_recolor_from_saved_arrays(self):
        scene, directory, manifest, split, protocol = self.fixture()
        path = self.root / "display.json"
        calibrate_display(scene, directory, split["fit"], path, split, protocol, bins=128, stride=1)
        source = self.root / "signal_evaluation"
        arrays = source / "validation/float_arrays"; arrays.mkdir(parents=True)
        q = np.load(directory / manifest["frames"]["hold"]["signal_file"])
        y = normalize_signal(q, protocol["signal_calibration"], 250, 450)[None]
        np.save(arrays / "hold.prediction.npy", y); np.save(arrays / "hold.gt.npy", y)
        (source / "metrics.json").write_text(json.dumps({"modality": "thermal", "observation_domain": "raw_rjpeg",
            "metric_domain": "full_frame_float_normalized_camera_signal", "radiometric_protocol": protocol,
            "splits": {"validation": {"views": 1}}}))
        (source / "validation/per_view.json").write_text(json.dumps([{"image_name": "hold", "PSNR": 120.,
            "SSIM": 1., "MSE": 0., "RMSE": 0., "MAE": 0., "camera_signal_MAE": 0.}]))
        output = self.root / "restored"
        result = recolor(scene, source, path, output, radiometric_dir=directory)
        summary = result["splits"]["validation"]
        self.assertGreater(summary["PSNR"], 35)
        self.assertEqual(summary["signal_PSNR"], 120.)
        self.assertAlmostEqual(summary["PSNR"], summary["mapping_roundtrip_PSNR"])
        for folder in ("renders", "gt", "comparisons", "roundtrip", "roundtrip_comparisons"):
            self.assertTrue((output / "validation" / folder / "hold.png").exists())
        with self.assertRaisesRegex(ValueError, "separate"):
            recolor(scene, source, path, source, radiometric_dir=directory)

    def test_rgb_metric_units_and_ssim_identity(self):
        a = np.full((24, 32, 3), .4); b = a + .1
        metrics = rgb_metrics(a, b)
        self.assertAlmostEqual(metrics["PSNR"], 20.)
        self.assertAlmostEqual(metrics["MAE"], .1)
        self.assertAlmostEqual(rgb_metrics(a, a)["SSIM"], 1.)

    def test_ssim_matches_repository_formula_when_torch_available(self):
        try:
            import torch
            import torch.nn.functional as F
            from torch.autograd import Variable
        except ImportError:
            self.skipTest("Torch is not installed")
        source = Path(__file__).resolve().parents[1] / "utils/loss_utils.py"
        tree = ast.parse(source.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("gaussian", "create_window", "ssim", "_ssim")]
        namespace = {"torch": torch, "F": F, "Variable": Variable, "exp": math.exp}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        rng = np.random.default_rng(9)
        a, b = rng.random((24, 32, 3)), rng.random((24, 32, 3))
        ta, tb = [torch.from_numpy(value.transpose(2, 0, 1).copy())[None].double() for value in (a, b)]
        torch_value = float(namespace["ssim"](ta, tb))
        self.assertAlmostEqual(rgb_metrics(a, b)["SSIM"], torch_value, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
