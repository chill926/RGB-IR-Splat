"""Decode RGBT-Scenes raw RJPEGs into float camera-equivalent signal maps.

Run before --observation_domain raw_rjpeg training. Does not modify the original
RGB, thermal, raw JPEGs, masks, or stage-1 geometry. Requires only NumPy/Pillow.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from utils.flir_radiometry import (CALIBRATION_KEYS, decode_flir_rjpeg,
    file_sha256, resize_signal, signal_to_apparent_temperature)
from utils.thermal_coordinates import (ALIGNED_COORDINATES, sample_native_signal,
    validate_profile, profile_sha256, validate_manifest_coordinates)


def prepare(scene, output, byte_order="auto", alignment=None):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if alignment is not None:
        validate_profile(alignment)
    for protected in (scene / "raw_images", scene / "rgb", scene / "thermal"):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Output must be separate from source image directories")
    if output == scene or output in scene.parents:
        raise ValueError("Output cannot replace or contain the scene directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output_dir")
    raw_files = {}
    for path in sorted((scene / "raw_images").iterdir()):
        if path.suffix.lower() in (".jpg", ".jpeg"):
            if path.stem in raw_files:
                raise ValueError("Ambiguous raw JPEG stem: " + path.stem)
            raw_files[path.stem] = path
    frames, calibration = {}, None
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        references = sorted(path for path in (scene / "thermal" / split).iterdir()
                            if path.suffix.lower() in extensions)
        if not references:
            raise ValueError("No thermal reference images in " + split)
        (output / split).mkdir(exist_ok=True)
        for reference in references:
            name = reference.stem
            if name in frames:
                raise ValueError("Duplicate train/test image stem: " + name)
            if name not in raw_files:
                raise FileNotFoundError("Missing radiometric raw JPEG: " + name)
            raw_path = raw_files[name]
            raw, metadata, diagnostics = decode_flir_rjpeg(raw_path, byte_order)
            coefficients = {key: metadata[key] for key in CALIBRATION_KEYS}
            if calibration is None:
                calibration = coefficients
            if coefficients != calibration:
                raise ValueError("Camera coefficients vary across frames; a single-response thermal field "
                                 "cannot safely mix these captures: " + name)
            with Image.open(reference) as image:
                size = image.size
                luma = np.asarray(image.convert("L"), dtype=np.float64)
            if abs(size[0] / size[1] - raw.shape[1] / raw.shape[0]) > 1e-6:
                raise ValueError("Raw/reference aspect ratios differ; supply registered signal maps instead: " + name)
            # This preserves the dataset thermal framing. It does not estimate
            # RGB/IR extrinsics or remove residual parallax.
            if alignment is None:
                signal = resize_signal(raw.astype(np.float64) + metadata["PlanckO"], size)
                valid = None
            else:
                if list(size) != alignment["reference_image_size"]:
                    raise ValueError("Frame size differs from alignment reference: " + name)
                if name in alignment["fit_references"]:
                    expected = alignment["fit_references"][name]
                    if (file_sha256(raw_path) != expected["raw_sha256"] or
                            file_sha256(reference) != expected["thermal_reference_sha256"]):
                        raise ValueError("Alignment fit images changed: " + name)
                signal, valid = sample_native_signal(raw.astype(np.float64) + metadata["PlanckO"], size, alignment)
            temperature = signal_to_apparent_temperature(signal, calibration)
            relative_path = f"{split}/{name}.npy"
            np.save(output / relative_path, signal, allow_pickle=False)
            correlation = None
            if signal.std() > 0 and luma.std() > 0:
                correlation = float(np.corrcoef(signal.reshape(-1), luma.reshape(-1))[0, 1])
            frames[name] = {
                "split": split, "signal_file": relative_path,
                "signal_sha256": file_sha256(output / relative_path),
                "raw_file": str(raw_path.relative_to(scene)),
                "raw_sha256": file_sha256(raw_path),
                "thermal_reference_sha256": file_sha256(reference),
                "shape": list(signal.shape), "native_shape": list(raw.shape),
                "signal_min": float(signal.min()), "signal_max": float(signal.max()),
                "apparent_temperature_K_percentiles": np.percentile(temperature, [1, 50, 99]).tolist(),
                "signal_vs_palette_luma_correlation": correlation,
                "camera_metadata": metadata, **diagnostics,
            }
            if valid is not None:
                mask_path = f"{split}/{name}.valid.npy"
                np.save(output / mask_path, valid, allow_pickle=False)
                frames[name].update(valid_mask_file=mask_path, valid_mask_sha256=file_sha256(output / mask_path),
                                    valid_pixel_fraction=float(valid.mean()))
    manifest = {
        "format_version": 1, "kind": "flir_rjpeg_camera_signal",
        "signal_definition": "Q = RawDN + PlanckO",
        "blackbody_response": "Q(T) = PlanckR1 / (PlanckR2 * (exp(PlanckB/T) - PlanckF))",
        "signal_calibration": calibration,
        "coordinate_system": "dataset_thermal_full_frame_resize",
        "resampling": "float32 bilinear in linear camera-signal domain",
        "source_scene": str(scene), "byte_order": byte_order,
        "surface_temperature_ground_truth": False,
        "corrections_applied": [],
        "notes": [
            "Camera-equivalent signal, not W/(m^2 sr) absolute band radiance.",
            "No per-frame min/max or histogram normalization is used for training.",
            "Camera emissivity/distance/air/window settings are recorded, not applied as surface corrections.",
            "Full-frame resize assumes raw and provided thermal images share framing; inspect overlays.",
            "Existing RGB/IR registration is retained; no RGB/IR pose or parallax correction is estimated.",
            "Palette correlation is a framing diagnostic, not a calibration or registration certificate.",
        ],
        "frames": frames,
        "apparent_temperature_extent_K": [
            float(signal_to_apparent_temperature(min(row["signal_min"] for row in frames.values()), calibration)),
            float(signal_to_apparent_temperature(max(row["signal_max"] for row in frames.values()), calibration)),
        ],
    }
    if alignment is not None:
        manifest.update(format_version=2, coordinate_system=ALIGNED_COORDINATES,
            coordinate_alignment=alignment, coordinate_alignment_sha256=profile_sha256(alignment),
            resampling="one-pass native float64 bilinear, half-pixel centres, float32 storage",
            corrections_applied=["fit_only_frozen_subpixel_translation"],
            validity_policy="border replicated for finite arrays; outside native FOV excluded by validity mask")
        validate_manifest_coordinates(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--scene", help="One RGBT-Scenes scene directory")
    sources.add_argument("--dataset_root", help="Convert every scene with raw_images under this dataset root")
    parser.add_argument("--output_dir", default="", help="Default: SCENE/radiometric. "
                        "In batch mode: optional parent directory for scene-specific outputs")
    parser.add_argument("--byte_order", choices=("auto", "native", "swap"), default="auto")
    parser.add_argument("--alignment", default="", help="Frozen fit-only coordinate profile, single-scene mode only")
    args = parser.parse_args()
    if args.alignment and args.dataset_root:
        raise ValueError("An alignment profile is scene-specific; do not apply one profile to a dataset batch")
    alignment = json.loads(Path(args.alignment).read_text()) if args.alignment else None
    scenes = ([Path(args.scene)] if args.scene else sorted(
        path for path in Path(args.dataset_root).iterdir() if path.is_dir() and (path / "raw_images").is_dir()))
    if not scenes:
        raise ValueError("No scene directories with raw_images were found")
    for scene in scenes:
        output = (Path(args.output_dir) / scene.name if args.dataset_root and args.output_dir
                  else Path(args.output_dir) if args.output_dir else scene / "radiometric")
        manifest = prepare(scene, output, args.byte_order, alignment=alignment)
        print(json.dumps({"scene": scene.name, "output_dir": str(output.resolve()),
                          "frames": len(manifest["frames"]),
                          "signal_calibration": manifest["signal_calibration"],
                          "apparent_temperature_extent_K": manifest["apparent_temperature_extent_K"],
                          "temperature_bounds_note": "Choose fixed model bounds covering these observations; "
                          "low emissivity may require a higher surface-temperature upper bound."}, indent=2), flush=True)


if __name__ == "__main__":
    main()
