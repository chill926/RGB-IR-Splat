"""Shared observation preparation for thermal training and evaluation."""
import json
from pathlib import Path

import torch

from utils.flir_radiometry import (CALIBRATION_KEYS, file_sha256, load_signal_frame,
    normalize_signal, resize_signal, validate_calibration)
from utils.thermal_physics import FlirCameraResponseLUT, UniformLWIRPlanckLUT


def radiometric_image_errors(prediction, target, response):
    """Image errors in signal and blackbody-equivalent K, not surface T errors."""
    span = response.physical_radiance_max - response.physical_radiance_min
    signal_error = (prediction - target) * span
    apparent_error = response.inverse(prediction) - response.inverse(target)
    return {"camera_signal_MAE": float(signal_error.abs().mean()),
            "camera_signal_RMSE": float(signal_error.square().mean().sqrt()),
            "apparent_temperature_MAE_K": float(apparent_error.abs().mean())}


def prepare_thermal_input(args, cameras, device):
    """Install normalized float radiometric targets on existing Camera objects.

    Legacy domains retain their existing targets and LUT. RJPEG inputs bypass
    pseudo-color grayscale, PNG quantization, and PIL-to-tensor clamping.
    Validation/test are decoded with fixed calibration; no statistics are fitted
    to them. Checkpoint provenance is verified before targets are replaced.
    """
    if args.observation_domain != "raw_rjpeg":
        return UniformLWIRPlanckLUT(args.temp_min, args.temp_max).to(device)
    directory = Path(getattr(args, "radiometric_dir", "") or "radiometric")
    if not directory.is_absolute():
        directory = Path(args.source_path) / directory
    manifest_path = directory / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (manifest.get("format_version") != 1 or manifest.get("kind") != "flir_rjpeg_camera_signal"
            or manifest.get("signal_definition") != "Q = RawDN + PlanckO"
            or manifest.get("coordinate_system") != "dataset_thermal_full_frame_resize"):
        raise ValueError("Unsupported radiometric manifest format/signal coordinates")
    calibration = validate_calibration(manifest["signal_calibration"])
    protocol = {
        "format_version": 1, "manifest_sha256": file_sha256(manifest_path),
        "signal_calibration": calibration,
        "normalization_temperature_bounds_K": [float(args.temp_min), float(args.temp_max)],
        "blackbody_response": "flir_camera_Q",
        "signal_definition": manifest["signal_definition"],
        "coordinate_system": manifest["coordinate_system"],
        "surface_temperature_ground_truth": False,
    }
    expected = getattr(args, "radiometric_protocol", None)
    if expected is not None and expected != protocol:
        raise ValueError("Radiometric calibration/normalization/manifest differs from the checkpoint")
    frames = manifest["frames"]
    prepared = []
    for camera in cameras:
        name = camera.image_name
        if name not in frames:
            raise ValueError("Missing radiometric observation for camera: " + name)
        record = frames[name]
        recorded = {key: float(record["camera_metadata"][key]) for key in CALIBRATION_KEYS}
        if recorded != calibration:
            raise ValueError("Frame camera response differs from scene calibration: " + name)
        signal = load_signal_frame(directory, record)
        target_size = (int(camera.image_width), int(camera.image_height))
        if abs(target_size[0] / target_size[1] - signal.shape[1] / signal.shape[0]) > 1e-6:
            raise ValueError("Camera/radiometric aspect ratios differ: " + name)
        # Validate native prepared values before downsampling: never hide an
        # out-of-range pixel through averaging or a silent clamp.
        normalize_signal(signal, calibration, args.temp_min, args.temp_max)
        if signal.shape != (target_size[1], target_size[0]):
            signal = resize_signal(signal, target_size)
        normalized = normalize_signal(signal, calibration, args.temp_min, args.temp_max)
        tensor = torch.from_numpy(normalized).unsqueeze(0).repeat(3, 1, 1)
        prepared.append((camera, tensor.to(device=device, dtype=torch.float32)))
    for camera, tensor in prepared:
        camera.original_physical_image = tensor
    args.radiometric_protocol = protocol
    # Keep relative paths portable when the entire scene moves to the server.
    args.radiometric_dir = getattr(args, "radiometric_dir", "") or "radiometric"
    print(f"[observation] raw_rjpeg: {len(prepared)} float signal maps; "
          "fixed camera calibration and shared temperature-bound normalization. "
          "Temperatures are model-based estimates, not surface-temperature GT.")
    return FlirCameraResponseLUT(calibration, args.temp_min, args.temp_max).to(device)
