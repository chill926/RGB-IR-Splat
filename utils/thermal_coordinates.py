"""Fit-only subpixel coordinate profiles and one-pass native-signal sampling.

These are empirical export-coordinate corrections, not RGB/IR camera calibration.
No OpenCV/Torch is required for estimation or data preparation.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, shift
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from utils.flir_radiometry import decode_flir_rjpeg, file_sha256, read_flir_display_metadata, resize_signal

LEGACY_COORDINATES = "dataset_thermal_full_frame_resize"
ALIGNED_COORDINATES = "dataset_thermal_fit_aligned"


def profile_sha256(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def validate_profile(profile):
    if (profile.get("format_version") != 1 or profile.get("kind") != "fit_only_signal_translation"
            or profile.get("sampling_convention") != "half_pixel_centres_native_to_target"
            or profile.get("fit_policy") != "fit_pixels_only"):
        raise ValueError("Unsupported signal alignment profile")
    size = profile.get("reference_image_size", [])
    offset = profile.get("offset_reference_pixels", [])
    if (len(size) != 2 or any(not isinstance(x, int) or x <= 1 for x in size)
            or len(offset) != 2 or not all(math.isfinite(float(x)) for x in offset)):
        raise ValueError("Invalid alignment size/offset")
    split = profile.get("camera_split", {})
    if not all(isinstance(split.get(k), list) and split[k] for k in ("fit", "validation", "test")):
        raise ValueError("Alignment must record the exact three camera splits")
    names = sum((split[k] for k in ("fit", "validation", "test")), [])
    if len(set(names)) != len(names) or profile.get("fit_camera_names") != split["fit"]:
        raise ValueError("Alignment splits overlap or fit names differ")
    refs = profile.get("fit_references", {})
    if set(refs) != set(split["fit"]):
        raise ValueError("Alignment must record all fit input hashes")
    if profile.get("held_out_pixels_used_for_fitting") is not False:
        raise ValueError("Alignment was not fitted exclusively on fit data")
    return profile


def validate_manifest_coordinates(manifest):
    version, system = manifest.get("format_version"), manifest.get("coordinate_system")
    if version == 1 and system == LEGACY_COORDINATES:
        return None
    if version != 2 or system != ALIGNED_COORDINATES:
        raise ValueError("Unsupported radiometric coordinate protocol")
    profile = validate_profile(manifest.get("coordinate_alignment", {}))
    if profile_sha256(profile) != manifest.get("coordinate_alignment_sha256"):
        raise ValueError("Alignment profile hash differs from manifest")
    names = sum((profile["camera_split"][k] for k in ("fit", "validation", "test")), [])
    if set(names) != set(manifest.get("frames", {})):
        raise ValueError("Alignment camera split differs from radiometric frames")
    for name, row in manifest["frames"].items():
        if row.get("shape") != list(reversed(profile["reference_image_size"])):
            raise ValueError("Prepared aligned frame differs from the reference image size")
        if not row.get("valid_mask_file") or not row.get("valid_mask_sha256"):
            raise ValueError("Aligned frames require hashed validity masks")
        expected_split = "test" if name in profile["camera_split"]["test"] else "train"
        if row.get("split") != expected_split:
            raise ValueError("Alignment/published split mismatch")
    for name, ref in profile["fit_references"].items():
        row = manifest["frames"][name]
        if ref["raw_sha256"] != row["raw_sha256"] or ref["thermal_reference_sha256"] != row["thermal_reference_sha256"]:
            raise ValueError("Alignment fit input hashes differ from prepared data")
    return profile


def sample_native_signal(signal, output_size, profile):
    """Target (x,y) samples unshifted target (x+dx,y+dy), in one native lookup.

    Resizing and correction are composed analytically. Border replication keeps
    arrays finite; the mask excludes locations outside the native field of view.
    """
    validate_profile(profile)
    source = np.asarray(signal, dtype=np.float64)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("Native signal must be a finite HxW array")
    width, height = (int(x) for x in output_size)
    if min(width, height) < 2:
        raise ValueError("Output signal size is too small")
    reference_w, reference_h = profile["reference_image_size"]
    dx, dy = profile["offset_reference_pixels"]
    native_h, native_w = source.shape
    x = (np.arange(width, dtype=np.float64) + .5) * native_w / width - .5 + dx * native_w / reference_w
    y = (np.arange(height, dtype=np.float64) + .5) * native_h / height - .5 + dy * native_h / reference_h
    yy, xx = np.meshgrid(y, x, indexing="ij")
    valid = (xx >= -.5) & (xx <= native_w - .5) & (yy >= -.5) & (yy <= native_h - .5)
    sampled = map_coordinates(source, [yy, xx], order=1, mode="nearest", prefilter=False).astype(np.float32)
    return sampled, valid


def load_valid_mask(directory, record):
    root = Path(directory).resolve()
    path = (root / record["valid_mask_file"]).resolve()
    if root not in path.parents or file_sha256(path) != record["valid_mask_sha256"]:
        raise ValueError("Invalid or changed radiometric validity mask")
    mask = np.load(path, allow_pickle=False)
    if mask.shape != tuple(record["shape"]) or mask.dtype != np.bool_:
        raise ValueError("Radiometric validity mask must be boolean and match the frame")
    return mask


def _thermal_path(scene, name, split="train"):
    matches = [p for p in (Path(scene) / "thermal" / split).iterdir()
               if p.stem == name and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous thermal image: " + name)
    return matches[0]


def estimate_translation(signal, palette_index, stride=4, max_shift=8):
    """Normalized correlation search; output offsets are target-image pixels."""
    q, target = np.asarray(signal, float)[::stride, ::stride], np.asarray(palette_index, float)[::stride, ::stride]
    lo, hi = np.percentile(q, [1, 99])
    if hi - lo <= 1e-8 or float(target.std()) <= 1e-6:
        raise ValueError("Alignment image has insufficient contrast")
    source = gaussian_filter(np.clip((q - lo) / (hi - lo), 0, 1), sigma=1)
    target = gaussian_filter(target, sigma=1)
    margin = int(math.ceil(max_shift / stride)) + 3
    if min(target.shape) <= 2 * margin + 4:
        raise ValueError("Alignment image is too small")
    template = target[margin:-margin, margin:-margin].reshape(-1)
    template = template - template.mean()
    norm = np.linalg.norm(template)
    def objective(offset):
        moved = shift(source, (-offset[1], -offset[0]), order=1, mode="nearest", prefilter=False)
        values = moved[margin:-margin, margin:-margin].reshape(-1)
        values = values - values.mean()
        return 1 - float(np.dot(values, template) / max(np.linalg.norm(values) * norm, 1e-15))
    limit = float(max_shift) / stride
    result = minimize(objective, [0., 0.], method="Powell", bounds=[(-limit, limit)] * 2,
                      options={"maxiter": 60, "xtol": 1e-4, "ftol": 1e-8})
    offset = result.x * stride
    before, after = 1 - objective([0, 0]), 1 - float(result.fun)
    accepted = (bool(result.success) and np.isfinite(offset).all() and after >= .9
                and after >= before - 1e-6 and np.max(np.abs(offset)) < max_shift * .98)
    return {"offset_pixels": offset.tolist(), "NCC_before": before, "NCC_after": after,
            "accepted": bool(accepted)}


def fit_alignment(scene, camera_split, stride=4, max_shift=8, byte_order="auto", progress=None):
    """Read pixel data only from fit; held-out images contribute filenames only."""
    from utils.thermal_pseudocolor import palette_rgb
    scene = Path(scene).resolve()
    if stride < 1 or not math.isfinite(max_shift) or not 0 < max_shift <= 16:
        raise ValueError("Invalid alignment sampling/shift limit")
    split = {k: list(camera_split.get(k, [])) for k in ("fit", "validation", "test")}
    names = sum(split.values(), [])
    if not all(split.values()) or len(set(names)) != len(names):
        raise ValueError("Alignment requires disjoint nonempty fit/validation/test splits")
    for role in ("fit", "validation", "test"):
        for name in split[role]:
            _thermal_path(scene, name, "test" if role == "test" else "train")
    raw_files = {p.stem: p for p in (scene / "raw_images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg")}
    rows, refs, reference_size = [], {}, None
    for name in split["fit"]:
        if name not in raw_files:
            raise ValueError("Missing fit raw JPEG: " + name)
        raw_path, target_path = raw_files[name], _thermal_path(scene, name)
        raw, metadata, _ = decode_flir_rjpeg(raw_path, byte_order)
        with Image.open(target_path) as image:
            target = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
            size = image.size
        if reference_size is None:
            reference_size = size
        if reference_size != size or abs(size[0]/size[1] - raw.shape[1]/raw.shape[0]) > 1e-6:
            raise ValueError("Alignment requires a consistent full-frame target coordinate system")
        signal = resize_signal(raw.astype(float) + metadata["PlanckO"], size)
        settings = read_flir_display_metadata(raw_path)
        candidates = []
        for encoding in ("full", "limited"):
            table = palette_rgb(settings["palette_ycrcb"], encoding)
            distance, index = cKDTree(table).query(target.reshape(-1, 3))
            candidates.append((float(np.mean(distance ** 2)), index / (len(table) - 1), encoding))
        _, index, encoding = min(candidates, key=lambda x: x[0])
        estimate = estimate_translation(signal, index.reshape(signal.shape), stride, max_shift)
        estimate.update(name=name, palette_encoding=encoding)
        rows.append(estimate)
        refs[name] = {"raw_sha256": file_sha256(raw_path), "thermal_reference_sha256": file_sha256(target_path)}
        if progress is not None:
            progress(len(rows), len(split["fit"]))
    accepted = [r["offset_pixels"] for r in rows if r["accepted"]]
    if len(accepted) < max(3, int(math.ceil(len(rows) * .5))):
        raise ValueError("Insufficient reliable fit frames for a global translation")
    median = np.median(accepted, axis=0)
    mad = np.median(np.abs(np.asarray(accepted) - median), axis=0)
    if np.max(mad) > 1:
        raise ValueError("Fit shifts vary too much for a fixed translation; inspect geometry/export instead")
    document = {"format_version": 1, "kind": "fit_only_signal_translation",
        "reference_image_size": list(reference_size), "offset_reference_pixels": median.tolist(),
        "sampling_convention": "half_pixel_centres_native_to_target", "fit_policy": "fit_pixels_only",
        "fit_camera_names": split["fit"], "camera_split": split, "fit_references": refs,
        "held_out_pixels_used_for_fitting": False, "estimator": "scipy_bounded_NCC_translation_median",
        "diagnostics": {"accepted_frames": len(accepted), "fit_frames": len(rows), "MAD_pixels": mad.tolist(),
                        "stride": stride, "max_shift_pixels": max_shift, "per_fit_frame": rows},
        "meaning": "empirical export-coordinate offset; not a recovered physical camera calibration"}
    return validate_profile(document)
