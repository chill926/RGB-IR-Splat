"""CPU tests for radiometric decoding, forward physics, and checkpoint inputs.

Run: python tools/test_thermal_radiometry.py
NumPy/Pillow tests always run. Torch/SciPy enable model and integration tests.
"""
import ast
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from utils.flir_radiometry import (blackbody_signal, decode_flir_rjpeg, file_sha256,
    load_signal_frame, normalize_signal, resize_signal, signal_to_apparent_temperature)
from tools.prepare_rgbt_radiometry import prepare

try:
    import torch
    from utils.thermal_physics import FlirCameraResponseLUT, MaterialThermalField
    from utils.thermal_observations import prepare_thermal_input, radiometric_image_errors
except ImportError:
    torch = None


CALIBRATIONS = [
    dict(PlanckR1=12487.9970703125, PlanckR2=0.02452719397842884,
         PlanckB=1345.5999755859375, PlanckF=1.600000023841858, PlanckO=-6726),
    dict(PlanckR1=16028.2587890625, PlanckR2=0.08414093405008316,
         PlanckB=1418.699951171875, PlanckF=1.149999976158142, PlanckO=-10430),
]


def rjpeg_bytes(raw, calibration, swapped=False, binary=False, endian="<"):
    height, width = raw.shape
    header = bytearray(32)
    struct.pack_into(endian + "HHH", header, 0, 2, width, height)
    stored = raw.byteswap() if swapped else raw
    if binary:
        payload = stored.astype(endian + "u2").tobytes()
    else:
        image = io.BytesIO()
        Image.fromarray(stored).save(image, format="PNG")
        payload = image.getvalue()
    raw_record = bytes(header) + payload
    camera = bytearray(0x340)
    struct.pack_into(endian + "H", camera, 0, 2)
    offsets = dict(PlanckR1=0x58, PlanckB=0x5C, PlanckF=0x60, PlanckR2=0x30C)
    for name, offset in offsets.items():
        struct.pack_into(endian + "f", camera, offset, calibration[name])
    struct.pack_into(endian + "i", camera, 0x308, calibration["PlanckO"])
    struct.pack_into(endian + "H", camera, 0x338, int(np.median(raw)))
    camera[0xD4:0xD4 + 11] = b"Test Camera"
    fff = bytearray(128)
    fff[:4] = b"FFF\x00"
    struct.pack_into(endian + "III", fff, 0x14, 100, 64, 2)
    struct.pack_into(endian + "H", fff, 64, 1)
    struct.pack_into(endian + "II", fff, 76, 128, len(raw_record))
    struct.pack_into(endian + "H", fff, 96, 0x20)
    struct.pack_into(endian + "II", fff, 108, 128 + len(raw_record), len(camera))
    fff = bytes(fff) + raw_record + bytes(camera)
    # Reversed APP1 storage order verifies sequence-number reassembly.
    pieces = [fff[:100], fff[100:]]
    jpeg = bytearray(b"\xff\xd8")
    for index in (1, 0):
        data = b"FLIR\x00\x01" + bytes([index, 1]) + pieces[index]
        jpeg += b"\xff\xe1" + struct.pack(">H", len(data) + 2) + data
    return bytes(jpeg) + b"\xff\xd9"


class RadiometryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = (12000 + np.arange(80).reshape(8, 10)).astype(np.uint16)

    def tearDown(self):
        self.temp.cleanup()

    def test_split_app1_native_and_swapped_png(self):
        for swapped in (False, True):
            path = self.root / "capture.jpg"
            path.write_bytes(rjpeg_bytes(self.raw, CALIBRATIONS[0], swapped=swapped))
            decoded, metadata, info = decode_flir_rjpeg(path)
            np.testing.assert_array_equal(decoded, self.raw)
            self.assertEqual(info["byteswap_used"], swapped)
            self.assertEqual(metadata["PlanckO"], -6726)

    def test_binary_big_and_little_endian(self):
        for endian in ("<", ">"):
            path = self.root / "capture.jpg"
            path.write_bytes(rjpeg_bytes(self.raw, CALIBRATIONS[1], binary=True, endian=endian))
            decoded, metadata, info = decode_flir_rjpeg(path)
            np.testing.assert_array_equal(decoded, self.raw)
            self.assertEqual(info["raw_encoding"], "binary16")

    def test_missing_and_truncated_payloads_fail(self):
        path = self.root / "capture.jpg"
        Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, "no FLIR"):
            decode_flir_rjpeg(path)
        path.write_bytes(rjpeg_bytes(self.raw, CALIBRATIONS[0])[:-100])
        with self.assertRaises(ValueError):
            decode_flir_rjpeg(path)

    def test_camera_response_roundtrip_and_fixed_normalization(self):
        for calibration in CALIBRATIONS:
            temperature = np.linspace(260, 950, 111)
            q = blackbody_signal(temperature, calibration)
            np.testing.assert_allclose(signal_to_apparent_temperature(q, calibration), temperature, atol=1e-9)
            a = normalize_signal(q, calibration, 250, 1000)
            b = normalize_signal(q[30:90], calibration, 250, 1000)
            np.testing.assert_array_equal(a[30:90], b)

    def test_linear_signal_mixing_survives_affine_normalization(self):
        for calibration in CALIBRATIONS:
            emission, environment = blackbody_signal([400, 290], calibration)
            epsilon = 0.3
            observation = epsilon * emission + (1 - epsilon) * environment
            normalized = normalize_signal(observation, calibration, 250, 1000)
            expected = epsilon * normalize_signal(emission, calibration, 250, 1000)
            expected += (1 - epsilon) * normalize_signal(environment, calibration, 250, 1000)
            self.assertAlmostEqual(float(normalized), float(expected), places=7)

    def test_out_of_range_is_not_silently_clipped(self):
        q = blackbody_signal([300, 700], CALIBRATIONS[1])
        with self.assertRaisesRegex(ValueError, "outside"):
            normalize_signal(q, CALIBRATIONS[1], 250, 450)

    def test_resize_preserves_float_signal_units(self):
        q = np.full((8, 10), 5432.125, dtype=np.float32)
        resized = resize_signal(q, (20, 16))
        self.assertEqual(resized.dtype, np.float32)
        np.testing.assert_array_equal(resized, np.full((16, 20), 5432.125, dtype=np.float32))

    def make_scene(self, calibration=CALIBRATIONS[0], varying=False):
        scene = self.root / "AnyScene"
        (scene / "raw_images").mkdir(parents=True)
        for index, split in enumerate(("train", "test")):
            (scene / "thermal" / split).mkdir(parents=True)
            name = f"arbitrary_{index}"
            raw = self.raw + index * 20
            coeff = CALIBRATIONS[1] if varying and index else calibration
            (scene / "raw_images" / f"{name}.jpg").write_bytes(rjpeg_bytes(raw, coeff, swapped=True))
            Image.fromarray(np.full((16, 20, 3), 100, dtype=np.uint8)).save(scene / "thermal" / split / f"{name}.png")
        return scene

    def test_generic_scene_manifest_and_content_integrity(self):
        scene = self.make_scene()
        output = self.root / "signals"
        manifest = prepare(scene, output)
        self.assertEqual(set(manifest["frames"]), {"arbitrary_0", "arbitrary_1"})
        row = manifest["frames"]["arbitrary_0"]
        np.testing.assert_allclose(load_signal_frame(output, row), resize_signal(self.raw.astype(float) - 6726, (20, 16)))
        np.save(output / row["signal_file"], np.zeros((16, 20), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "changed"):
            load_signal_frame(output, row)

    def test_manifest_path_cannot_escape_output(self):
        path = self.root / "outside.npy"
        np.save(path, np.ones((2, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "escapes"):
            load_signal_frame(self.root / "signals", dict(signal_file="../outside.npy", signal_sha256=file_sha256(path), shape=[2, 2]))

    def test_mixed_camera_responses_rejected(self):
        scene = self.make_scene(varying=True)
        with self.assertRaisesRegex(ValueError, "coefficients vary"):
            prepare(scene, self.root / "signals")

    @unittest.skipIf(torch is None, "Torch/SciPy are not installed")
    def test_response_lut_inverse_gradient_and_emission_reflection(self):
        for calibration in CALIBRATIONS:
            response = FlirCameraResponseLUT(calibration, 250, 1000)
            temperature = torch.tensor([280., 300., 500., 800.], requires_grad=True)
            output = response(temperature)
            expected = normalize_signal(blackbody_signal(temperature.detach().numpy(), calibration), calibration, 250, 1000)
            np.testing.assert_allclose(output.detach().numpy(), expected, atol=1e-7)
            torch.testing.assert_close(response.inverse(output), temperature, atol=1e-3, rtol=0)
            output.sum().backward()
            self.assertTrue(bool(torch.isfinite(temperature.grad).all()))
            self.assertTrue(bool((temperature.grad > 0).all()))
            field = MaterialThermalField(torch.tensor([0]), torch.tensor([[.2]]), [0.3],
                                         temp_min=250, temp_max=1000, initial_environment=.02)
            expected_field = .3 * response(field.temperature) + .7 * field.environment
            torch.testing.assert_close(field.radiance(response), expected_field)
            field.radiance(response).sum().backward()
            self.assertTrue(bool(torch.isfinite(field.temperature_raw.grad).all()))

    @unittest.skipIf(torch is None, "Torch/SciPy are not installed")
    def test_camera_targets_and_checkpoint_protocol_roundtrip(self):
        scene = self.make_scene()
        prepare(scene, scene / "radiometric")
        args = SimpleNamespace(source_path=str(scene), observation_domain="raw_rjpeg", temp_min=250, temp_max=450)
        camera = SimpleNamespace(image_name="arbitrary_0", image_width=20, image_height=16)
        response = prepare_thermal_input(args, [camera], "cpu")
        self.assertEqual(camera.original_physical_image.shape, (3, 16, 20))
        protocol = json.loads(json.dumps(args.radiometric_protocol))
        reconstructed = SimpleNamespace(**vars(args))
        reconstructed.radiometric_protocol = protocol
        camera2 = SimpleNamespace(image_name="arbitrary_0", image_width=10, image_height=8)
        other_response = prepare_thermal_input(reconstructed, [camera2], "cpu")
        torch.testing.assert_close(response.radiance, other_response.radiance)
        reconstructed.temp_max = 500
        with self.assertRaisesRegex(ValueError, "differs from the checkpoint"):
            prepare_thermal_input(reconstructed, [camera2], "cpu")

    @unittest.skipIf(torch is None, "Torch/SciPy are not installed")
    def test_train_validation_and_evaluation_conversion_contract(self):
        # Load the real conversion function without requiring CUDA rasterizer
        # imports. Feed the same prepared targets used by all three consumers.
        source = Path(__file__).resolve().parents[1] / "train_thermal_physics.py"
        tree = ast.parse(source.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "observation_to_model_radiance")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        function = namespace[node.name]
        target = torch.tensor([.01, .10, .50])
        self.assertIs(function(target, SimpleNamespace(observation_domain="raw_rjpeg"), None), target)
        legacy = function(torch.tensor([-1., .5, 2.]), SimpleNamespace(observation_domain="normalized_dn"), None)
        torch.testing.assert_close(legacy, torch.tensor([0., .5, 1.]))
        with self.assertRaisesRegex(ValueError, "Invalid"):
            function(torch.tensor([float("nan")]), SimpleNamespace(observation_domain="raw_rjpeg"), None)

    @unittest.skipIf(torch is None, "Torch/SciPy are not installed")
    def test_signal_and_apparent_temperature_error_units(self):
        response = FlirCameraResponseLUT(CALIBRATIONS[1], 250, 1000)
        prediction, target = response(torch.tensor([510.])), response(torch.tensor([500.]))
        errors = radiometric_image_errors(prediction, target, response)
        self.assertAlmostEqual(errors["apparent_temperature_MAE_K"], 10., places=3)
        expected_signal_error = float(blackbody_signal(510, CALIBRATIONS[1]) - blackbody_signal(500, CALIBRATIONS[1]))
        self.assertAlmostEqual(errors["camera_signal_MAE"], expected_signal_error, delta=.01)

    @unittest.skipIf(torch is None, "Torch/SciPy are not installed")
    def test_branch_inherits_and_requires_calibration_protocol(self):
        source = Path(__file__).resolve().parents[1] / "train_thermal_physics.py"
        tree = ast.parse(source.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "apply_stage2_protocol")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        function = namespace[node.name]
        checkpoint_path = self.root / "stage2.pt"
        shared = {"observation_domain": "raw_rjpeg", "temp_min": 250., "temp_max": 1000.,
                  "radiometric_dir": "radiometric", "radiometric_protocol": {"manifest_sha256": "example"}}
        torch.save({"metadata": {"shared_training_protocol": shared}}, checkpoint_path)
        args = SimpleNamespace(stage="branch", stage2_checkpoint=str(checkpoint_path),
                               radiometric_dir="", environment_lr=1e-4)
        function(args)
        self.assertEqual(args.observation_domain, "raw_rjpeg")
        self.assertEqual(args.temp_max, 1000.)
        self.assertEqual(args.radiometric_protocol, shared["radiometric_protocol"])
        self.assertEqual(args.radiometric_dir, "radiometric")
        shared.pop("radiometric_protocol")
        torch.save({"metadata": {"shared_training_protocol": shared}}, checkpoint_path)
        with self.assertRaisesRegex(ValueError, "missing"):
            function(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
