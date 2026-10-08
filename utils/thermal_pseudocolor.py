"""Frozen, train-calibrated FLIR display mapping and original-RGB evaluation.

No predicted image or held-out reference image is used to fit this mapping.
Known per-capture window metadata is allowed; palettes and response curves are
frozen. This is an estimated export/display response, not guaranteed exact AGC.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.spatial import cKDTree

from utils.flir_radiometry import (blackbody_signal, file_sha256, load_signal_frame,
    read_flir_display_metadata, validate_calibration)


def inside_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Display input path escapes its directory")
    return path


def load_manifest(scene, directory="radiometric", expected=None):
    directory = Path(directory or "radiometric")
    if not directory.is_absolute():
        directory = Path(scene) / directory
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("format_version") != 1 or manifest.get("kind") != "flir_rjpeg_camera_signal"
            or manifest.get("signal_definition") != "Q = RawDN + PlanckO"
            or manifest.get("coordinate_system") != "dataset_thermal_full_frame_resize"):
        raise ValueError("Pseudo-color requires prepared FLIR camera-signal inputs")
    digest = file_sha256(path)
    calibration = validate_calibration(manifest["signal_calibration"])
    if expected is not None and (digest != expected["manifest_sha256"]
                                or calibration != expected["signal_calibration"]):
        raise ValueError("Radiometric manifest/calibration differs from the checkpoint")
    return directory.resolve(), manifest, digest


def thermal_reference(scene, name, record):
    directory = Path(scene) / "thermal" / record["split"]
    matches = sorted(p for p in directory.iterdir()
                     if p.stem == name and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous original thermal reference: " + name)
    if file_sha256(matches[0]) != record["thermal_reference_sha256"]:
        raise ValueError("Original thermal reference changed since radiometric preparation: " + name)
    return matches[0]


def original_rgb(scene, name, record, shape=None):
    with Image.open(thermal_reference(scene, name, record)) as image:
        image = image.convert("RGB")
        if shape is not None and image.size != (shape[1], shape[0]):
            # Match the repository's PILtoTorch RGB resizing policy.
            image = image.resize((shape[1], shape[0]))
        return np.asarray(image, dtype=np.float32) / 255.0


def palette_rgb(ycrcb, encoding):
    table = np.asarray(ycrcb, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 3 or len(table) < 2:
        raise ValueError("Invalid FLIR YCrCb palette")
    y, cr, cb = table[:, 0], table[:, 1] - 128, table[:, 2] - 128
    if encoding == "limited":
        y = (y - 16) * 255 / 219
        cr, cb = cr * 255 / 224, cb * 255 / 224
    elif encoding != "full":
        raise ValueError("Palette encoding must be full or limited")
    rgb = np.stack((y + 1.402 * cr, y - .714136 * cr - .344136 * cb, y + 1.772 * cb), axis=1)
    return (np.clip(rgb, 0, 255) / 255).astype(np.float32)


def isotonic_response(values, weights):
    """Weighted pool-adjacent-violators fit; increasing signal => increasing index."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.shape != weights.shape or values.ndim != 1 or not len(values):
        raise ValueError("Invalid isotonic response arrays")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Isotonic weights must be positive and finite")
    levels, masses, starts, ends = [], [], [], []
    for index, (value, weight) in enumerate(zip(values, weights)):
        levels.append(float(value)); masses.append(float(weight)); starts.append(index); ends.append(index + 1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            value = (levels[-2] * masses[-2] + levels[-1] * masses[-1]) / (masses[-2] + masses[-1])
            levels[-2:] = [value]; masses[-2:] = [masses[-2] + masses[-1]]
            ends[-2:] = [ends[-1]]; starts.pop()
    result = np.empty_like(values)
    for start, end, value in zip(starts, ends, levels):
        result[start:end] = value
    return result


def fit_response(window_coordinate, palette_index, bins=512):
    z, y = np.asarray(window_coordinate, float), np.asarray(palette_index, float)
    if z.shape != y.shape or z.ndim != 1 or len(z) < 32 or not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("Insufficient/invalid display calibration samples")
    lo, hi = float(z.min()), float(z.max())
    if not lo < hi or bins < 8:
        raise ValueError("Display calibration needs signal variation and at least 8 bins")
    index = np.minimum(((z - lo) / (hi - lo) * bins).astype(int), bins - 1)
    count = np.bincount(index, minlength=bins)
    sum_z = np.bincount(index, weights=z, minlength=bins)
    sum_y = np.bincount(index, weights=y, minlength=bins)
    valid = count > 0
    x = sum_z[valid] / count[valid]
    u = isotonic_response(sum_y[valid] / count[valid], count[valid])
    if len(x) < 2 or not np.all(np.diff(x) > 0):
        raise ValueError("Display calibration has insufficient independent signal samples")
    # Continuous extrapolation to palette endpoints for signals outside support.
    # These tails are flagged during evaluation; they are not silently rescaled.
    edge = max(1, min(len(x) - 1, len(x) // 20))
    global_slope = max(float((u[-1] - u[0]) / (x[-1] - x[0])), 1e-6)
    left = float((u[edge] - u[0]) / (x[edge] - x[0]))
    right = float((u[-1] - u[-1-edge]) / (x[-1] - x[-1-edge]))
    return {"window_knots": x.tolist(), "palette_fraction_knots": np.clip(u, 0, 1).tolist(),
            "support": [lo, hi], "tail_slopes": [left or global_slope, right or global_slope],
            "samples": int(len(z)), "bins": int(bins)}


def colorize_signal(signal, display, group):
    q = np.asarray(signal, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all():
        raise ValueError("Display input must be a finite HxW camera-signal image")
    z = (q - display["signal_window_center"]) / display["raw_value_range"] + .5
    curve = group["response"]
    x, y = np.asarray(curve["window_knots"]), np.asarray(curve["palette_fraction_knots"])
    index = np.interp(z, x, y)
    index = np.where(z < x[0], y[0] + (z - x[0]) * curve["tail_slopes"][0], index)
    index = np.where(z > x[-1], y[-1] + (z - x[-1]) * curve["tail_slopes"][1], index)
    index = np.clip(index, 0, 1)
    table = np.asarray(group["palette_rgb"], dtype=np.float32)
    palette_x = np.linspace(0, 1, len(table))
    rgb = np.stack([np.interp(index, palette_x, table[:, channel]) for channel in range(3)], axis=-1)
    lo, hi = curve["support"]
    return rgb.astype(np.float32), float(((z < lo) | (z > hi)).mean())


def normalized_to_signal(image, calibration, bounds):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2 or not np.isfinite(image).all() or image.min() < -1e-6 or image.max() > 1 + 1e-6:
        raise ValueError("Expected a finite normalized 1xHxW or HxW signal render")
    lo, hi = blackbody_signal(bounds, calibration)
    return image.astype(np.float64) * (hi - lo) + lo


def rgb_metrics(prediction, target):
    """RGB PSNR and SSIM matching the repository's 11px Gaussian/zero-pad policy."""
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("RGB metrics require equal HxWx3 images")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite RGB metric input")
    if min(prediction.min(), target.min()) < -1e-6 or max(prediction.max(), target.max()) > 1 + 1e-6:
        raise ValueError("RGB metric data range must be [0,1]")
    error = prediction - target
    mse = float(np.mean(error * error))
    kernel = np.exp(-np.arange(-5, 6, dtype=float) ** 2 / (2 * 1.5 ** 2))
    kernel /= kernel.sum()
    def blur(value):
        return convolve1d(convolve1d(value, kernel, axis=0, mode="constant", cval=0),
                          kernel, axis=1, mode="constant", cval=0)
    mu1, mu2 = blur(prediction), blur(target)
    var1, var2 = blur(prediction * prediction) - mu1 * mu1, blur(target * target) - mu2 * mu2
    covariance = blur(prediction * target) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + .01 ** 2) * (2 * covariance + .03 ** 2) /
            ((mu1 * mu1 + mu2 * mu2 + .01 ** 2) * (var1 + var2 + .03 ** 2)))
    return {"PSNR": -10 * math.log10(max(mse, 1e-12)), "SSIM": float(ssim.mean()),
            "MSE": mse, "RMSE": math.sqrt(mse), "MAE": float(np.mean(np.abs(error)))}


def _frame_display(scene, name, record):
    path = inside_path(scene, record["raw_file"])
    if file_sha256(path) != record["raw_sha256"]:
        raise ValueError("Raw JPEG changed since preparation: " + name)
    return read_flir_display_metadata(path)


def calibrate_display(scene, directory, fit_names, output, camera_split=None,
                      expected_protocol=None, bins=512, stride=4):
    """Fit only supplied fit views; held-out files contribute metadata, no pixels."""
    scene, output = Path(scene).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Display calibration already exists; reuse it or choose a new path")
    directory, manifest, digest = load_manifest(scene, directory, expected_protocol)
    names = list(fit_names)
    if not names or len(set(names)) != len(names) or stride < 1:
        raise ValueError("Unique fit-camera names and a positive stride are required")
    for name in names:
        if name not in manifest["frames"] or manifest["frames"][name]["split"] != "train":
            raise ValueError("Display fitting can only use published training cameras: " + name)
    if camera_split is not None:
        if names != camera_split.get("fit") or set(names) & (set(camera_split.get("validation", [])) | set(camera_split.get("test", []))):
            raise ValueError("Display fitting must use exactly the checkpoint fit split")
    displays = {name: _frame_display(scene, name, record)
                for name, record in manifest["frames"].items()}
    groups = {}
    for name in names:
        display = displays[name]
        key = display["palette_sha256"]
        group = groups.setdefault(key, {"fit_names": [], "palette_ycrcb": display["palette_ycrcb"]})
        group["fit_names"].append(name)
    for key, group in groups.items():
        samples, coordinates = [], []
        for name in group["fit_names"]:
            record, display = manifest["frames"][name], displays[name]
            signal = load_signal_frame(directory, record)[::stride, ::stride]
            target = original_rgb(scene, name, record)[::stride, ::stride]
            if target.shape[:2] != signal.shape:
                raise ValueError("Signal/original thermal dimensions differ: " + name)
            samples.append(target.reshape(-1, 3))
            coordinates.append(((signal - display["signal_window_center"]) /
                                display["raw_value_range"] + .5).reshape(-1))
        samples, coordinates = np.concatenate(samples), np.concatenate(coordinates)
        candidates = {}
        for encoding in ("limited", "full"):
            table = palette_rgb(group["palette_ycrcb"], encoding)
            distance, index = cKDTree(table).query(samples)
            candidates[encoding] = (float(np.mean(distance ** 2) / 3), table, index)
        encoding = min(candidates, key=lambda item: candidates[item][0])
        projection_mse, table, index = candidates[encoding]
        group.update({"palette_encoding": encoding, "palette_rgb": table.tolist(),
                      "palette_projection_mse_fit": projection_mse,
                      "response_source_palette": key,
                      "calibration_support": "fit_pixels",
                      "response": fit_response(coordinates, index / (len(table) - 1), bins)})
        del group["palette_ycrcb"]
    # A palette present only in a held-out capture is still known camera
    # metadata. Apply that palette with the largest training group's frozen
    # normalized response/encoding; never fit it to held-out colors. This
    # fallback is explicitly recorded and its roundtrip error is reported.
    primary = max(groups, key=lambda key: len(groups[key]["fit_names"]))
    for display in displays.values():
        key = display["palette_sha256"]
        if key not in groups:
            donor = groups[primary]
            groups[key] = {"fit_names": [], "palette_encoding": donor["palette_encoding"],
                           "palette_rgb": palette_rgb(display["palette_ycrcb"], donor["palette_encoding"]).tolist(),
                           "calibration_support": "metadata_palette_with_shared_fit_response",
                           "response_source_palette": primary, "response": donor["response"]}
    # Do not serialize the repeated native palette. Frozen RGB palettes reside
    # in groups; per-frame metadata only selects/conditions an existing mapping.
    for display in displays.values():
        display.pop("palette_ycrcb")
    document = {"format_version": 1, "kind": "flir_train_calibrated_pseudocolor",
                "signal_definition": "Q = RawDN + PlanckO", "manifest_sha256": digest,
                "signal_calibration": validate_calibration(manifest["signal_calibration"]),
                "fit_camera_names": names, "reference_camera_split": camera_split,
                "fit_policy": "Only checkpoint fit pixels; no validation/test color or signal statistics fitted",
                "display_model": "known palette/window metadata + frozen monotone scene response",
                "exact_camera_agc_reproduction": False,
                "bins": int(bins), "sample_stride": int(stride),
                "groups": groups, "frame_display_metadata": displays,
                "thermal_reference_sha256": {name: row["thermal_reference_sha256"]
                                               for name, row in manifest["frames"].items()},
                "source_scene": str(scene)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return document


class FrozenDisplay:
    def __init__(self, path, expected_protocol=None, fit_names=None):
        self.path = Path(path)
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        doc = self.document
        if doc.get("format_version") != 1 or doc.get("kind") != "flir_train_calibrated_pseudocolor":
            raise ValueError("Unsupported pseudo-color calibration")
        if expected_protocol is not None and (doc["manifest_sha256"] != expected_protocol["manifest_sha256"]
                                             or doc["signal_calibration"] != expected_protocol["signal_calibration"]):
            raise ValueError("Display/radiometric input provenance differs")
        if fit_names is not None and list(fit_names) != doc["fit_camera_names"]:
            raise ValueError("Display calibration did not use this checkpoint's exact fit cameras")
        for group in doc["groups"].values():
            curve = group["response"]
            x, y = np.asarray(curve["window_knots"]), np.asarray(curve["palette_fraction_knots"])
            if (len(x) < 2 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not np.all(np.diff(x) > 0) or np.any(np.diff(y) < 0) or y.min() < 0 or y.max() > 1):
                raise ValueError("Invalid frozen display response")
            table = np.asarray(group["palette_rgb"])
            if table.ndim != 2 or table.shape[1] != 3 or not np.isfinite(table).all() or table.min() < 0 or table.max() > 1:
                raise ValueError("Invalid frozen RGB palette")
            slopes = np.asarray(curve["tail_slopes"])
            if slopes.shape != (2,) or not np.isfinite(slopes).all() or np.any(slopes <= 0):
                raise ValueError("Invalid display extrapolation slopes")
        self.sha256 = file_sha256(self.path)

    def colorize(self, normalized, name, bounds):
        if name not in self.document["frame_display_metadata"]:
            raise ValueError("No known display settings for view: " + name)
        display = self.document["frame_display_metadata"][name]
        signal = normalized_to_signal(normalized, self.document["signal_calibration"], bounds)
        return colorize_signal(signal, display, self.document["groups"][display["palette_sha256"]])

    def colorize_with_settings(self, normalized, bounds, settings):
        """Render an arbitrary view with known display settings, without a GT image."""
        key = settings["palette_sha256"]
        if key not in self.document["groups"]:
            raise ValueError("The requested display palette has no training calibration")
        signal = normalized_to_signal(normalized, self.document["signal_calibration"], bounds)
        return colorize_signal(signal, settings, self.document["groups"][key])

    def verify_reference(self, name, record):
        if self.document["thermal_reference_sha256"].get(name) != record["thermal_reference_sha256"]:
            raise ValueError("Display reference provenance differs: " + name)


def evaluate_pseudocolor(display, normalized_prediction, normalized_target, original_target,
                        name, bounds):
    prediction, outside = display.colorize(normalized_prediction, name, bounds)
    roundtrip, observed_outside = display.colorize(normalized_target, name, bounds)
    metrics = rgb_metrics(prediction, original_target)
    metrics.update({"mapping_roundtrip_" + key: value
                    for key, value in rgb_metrics(roundtrip, original_target).items()})
    metrics.update({"prediction_outside_display_support_fraction": outside,
                    "observation_outside_display_support_fraction": observed_outside})
    return prediction, roundtrip, metrics


def save_rgb(path, image):
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)
