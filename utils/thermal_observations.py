"""Shared observation preparation for thermal training and evaluation."""
import json
from pathlib import Path

import torch

from utils.flir_radiometry import (CALIBRATION_KEYS, file_sha256, load_signal_frame,
    normalize_signal, resize_signal, validate_calibration, decode_flir_rjpeg)
from utils.thermal_coordinates import validate_manifest_coordinates, load_valid_mask, sample_native_signal
from utils.thermal_training_utils import masked_mean
from utils.thermal_physics import FlirCameraResponseLUT, UniformLWIRPlanckLUT


def radiometric_image_errors(prediction, target, response, mask=None):
    """Image errors in signal and blackbody-equivalent K, not surface T errors."""
    span = response.physical_radiance_max - response.physical_radiance_min
    signal_error = (prediction - target) * span
    apparent_error = response.inverse(prediction) - response.inverse(target)
    return {"camera_signal_MAE": float(masked_mean(signal_error.abs(), mask)),
            "camera_signal_RMSE": float(masked_mean(signal_error.square(), mask).sqrt()),
            "apparent_temperature_MAE_K": float(masked_mean(apparent_error.abs(), mask))}


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
    if (manifest.get("kind") != "flir_rjpeg_camera_signal"
            or manifest.get("signal_definition") != "Q = RawDN + PlanckO"):
        raise ValueError("Unsupported radiometric manifest format/signal coordinates")
    alignment = validate_manifest_coordinates(manifest)
    fit_names = getattr(args, "fit_camera_names", None)
    if alignment is not None and fit_names is not None and list(fit_names) != alignment["fit_camera_names"]:
        raise ValueError("Alignment was not fitted on this experiment's exact fit split")
    calibration = validate_calibration(manifest["signal_calibration"])
    protocol = {
        "format_version": manifest["format_version"], "manifest_sha256": file_sha256(manifest_path),
        "signal_calibration": calibration,
        "normalization_temperature_bounds_K": [float(args.temp_min), float(args.temp_max)],
        "blackbody_response": "flir_camera_Q",
        "signal_definition": manifest["signal_definition"],
        "coordinate_system": manifest["coordinate_system"],
        "surface_temperature_ground_truth": False,
    }
    if alignment is not None:
        protocol.update(coordinate_alignment_sha256=manifest["coordinate_alignment_sha256"],
            coordinate_fit_camera_names=alignment["fit_camera_names"], signal_metric_region="valid_native_FOV")
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
        valid = load_valid_mask(directory, record) if alignment is not None else None
        target_size = (int(camera.image_width), int(camera.image_height))
        if abs(target_size[0] / target_size[1] - signal.shape[1] / signal.shape[0]) > 1e-6:
            raise ValueError("Camera/radiometric aspect ratios differ: " + name)
        # Validate native prepared values before downsampling: never hide an
        # out-of-range pixel through averaging or a silent clamp.
        normalize_signal(signal, calibration, args.temp_min, args.temp_max)
        if signal.shape != (target_size[1], target_size[0]):
            if alignment is None:
                signal = resize_signal(signal, target_size)
            else:
                # Do not downsample an already interpolated/corrected signal.
                # Compose the requested resolution with the frozen native map.
                scene_root = Path(args.source_path).resolve()
                raw_path = (scene_root / record["raw_file"]).resolve()
                if scene_root not in raw_path.parents or file_sha256(raw_path) != record["raw_sha256"]:
                    raise ValueError("Native raw JPEG differs from aligned data")
                raw, metadata, _ = decode_flir_rjpeg(raw_path, manifest["byte_order"])
                signal, valid = sample_native_signal(raw.astype(float) + metadata["PlanckO"], target_size, alignment)
        normalized = normalize_signal(signal, calibration, args.temp_min, args.temp_max)
        tensor = torch.from_numpy(normalized).unsqueeze(0).repeat(3, 1, 1)
        mask = torch.from_numpy(valid).unsqueeze(0).to(device) if valid is not None else None
        prepared.append((camera, tensor.to(device=device, dtype=torch.float32), mask))
    for camera, tensor, mask in prepared:
        camera.original_physical_image = tensor
        camera.thermal_valid_mask = mask
    args.radiometric_protocol = protocol
    # Keep relative paths portable when the entire scene moves to the server.
    args.radiometric_dir = getattr(args, "radiometric_dir", "") or "radiometric"
    print(f"[observation] raw_rjpeg: {len(prepared)} float signal maps; "
          "fixed camera calibration and shared temperature-bound normalization. "
          "Temperatures are model-based estimates, not surface-temperature GT.")
    return FlirCameraResponseLUT(calibration, args.temp_min, args.temp_max).to(device)
