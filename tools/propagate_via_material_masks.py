"""Propagate masks exported by json_to_mask.py using the project's SAM2 library.

This is inference, not SAM2 training. Manual keyframes partition the sequence into
nearest-keyframe intervals; object IDs are local to an interval. Region indices
are never mistaken for persistent cross-keyframe identities. No SAM2 core edits.
Run only on the server with numpy, Pillow, torch, and SAM2 installed.
Material classes are read from class_map.json; no scene-specific class list.
--validate_only checks masks and priors without loading torch, SAM2 or weights.
"""
import argparse
import colorsys
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
from PIL import Image


UNKNOWN = 255
PRESET_COLORS = {"paint": (235, 70, 70), "plastic": (245, 170, 40),
          "rubber": (160, 85, 210), "metal": (45, 190, 100),
          "glass": (40, 160, 240)}
SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.lower())
                 for part in re.split(r"(\d+)", path.name))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def material_name(value):
    if not isinstance(value, str):
        raise ValueError("Material names must be text.")
    name = value.strip().lower()
    return "unknown" if name in {"", "unknown", "ignore"} else name


def material_color(name):
    if name in PRESET_COLORS:
        return list(PRESET_COLORS[name])
    hue = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big") / 2**32
    return [round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.72, 0.95)]


def read_material_schema(document, records):
    rows = document.get("classes", document.get("materials"))
    if not isinstance(rows, list):
        raise ValueError("class_map.json needs a classes or materials list.")
    unknown_id = document.get("unknown_id", document.get("unknown_label", 255))
    def valid_id(value):
        return not isinstance(value, bool) and isinstance(value, int) and 0 <= value <= 255
    if not valid_id(unknown_id):
        raise ValueError("Unknown source ID must be an integer in [0, 255].")
    source_ids, colors, seen_ids, seen_names = {}, {}, set(), set()
    for row in rows:
        name, source_id = material_name(row["name"]), row["id"]
        if not valid_id(source_id) or source_id in seen_ids or name in seen_names:
            raise ValueError("Source class names/IDs must be unique uint8 values.")
        seen_ids.add(source_id)
        seen_names.add(name)
        if name == "unknown":
            if source_id != unknown_id:
                raise ValueError("Unknown row ID differs from unknown_id/unknown_label.")
            continue
        if source_id == unknown_id:
            raise ValueError("A material cannot use the unknown source ID.")
        source_ids[name] = source_id
        color = row.get("color_rgb", material_color(name))
        if (not isinstance(color, (list, tuple)) or len(color) != 3 or
                any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= 255 for c in color)):
            raise ValueError("color_rgb needs three integers in [0, 255].")
        colors[name] = list(color)
    if not source_ids or len(source_ids) > 255:
        raise ValueError("Expected 1..255 source material classes.")
    epsilons = {}
    for record in records.values():
        for region in record.get("regions", []):
            name = material_name(region["material"])
            if name == "unknown":
                continue
            if name not in source_ids:
                raise ValueError("Manifest material absent from class_map.json: " + name)
            values = epsilons.setdefault(name, set())
            epsilon = region.get("epsilon_assumed")
            if epsilon is not None:
                value = float(epsilon)
                if not math.isfinite(value) or not 0.01 <= value <= 0.99:
                    raise ValueError("Stage-2 epsilon must lie in [0.01, 0.99]: " + name)
                values.add(value)
    invalid = {name: sorted(values) for name, values in epsilons.items() if len(values) != 1}
    if invalid:
        raise ValueError("Each observed material needs one consistent epsilon assumption: " + str(invalid))
    if not epsilons:
        raise ValueError("Manifest contains no material-labelled regions.")
    # Ignore unobserved classes in a reusable global map; no invented priors.
    names = sorted(epsilons, key=source_ids.__getitem__)
    output_ids = {name: index for index, name in enumerate(names)}
    epsilon_by_id = {output_ids[name]: next(iter(epsilons[name])) for name in names}
    return source_ids, unknown_id, output_ids, colors, epsilon_by_id


def material_rows(output_ids, epsilon_by_id, learned_materials):
    learned = {material_name(name) for name in learned_materials}
    absent = learned - set(output_ids)
    if absent:
        raise ValueError("--learn_materials names absent from this scene: " + ", ".join(sorted(absent)))
    return [{"id": class_id, "name": name, "epsilon0": epsilon_by_id[class_id],
             "confirmed": True, "source": "User-entered VIA epsilon assumption",
             "surface_condition": "Manual image-based label; not measured emissivity",
             "learn_k": name in learned, "learn_delta": name in learned,
             "k_prior": 0.0, "sigma_k": 0.0001, "sigma_delta": 0.05}
            for name, class_id in output_ids.items()]


def relative_file(root, filename):
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes its directory: " + str(filename))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_label(path):
    with Image.open(path) as image:
        value = np.asarray(image).copy()
    if value.ndim != 2:
        raise ValueError("Expected single-channel mask: " + str(path))
    return value


def load_keyframe(record, mask_root, source_ids, unknown_id, dimensions, output_ids):
    semantic = load_label(relative_file(mask_root, record["material_mask"]))
    if semantic.shape != dimensions:
        raise ValueError(record["filename"] + ": mask/RGB dimensions differ")
    valid_ids = list(source_ids.values()) + [unknown_id]
    if not np.isin(semantic, valid_ids).all():
        raise ValueError(record["filename"] + ": mask contains unmapped class IDs")
    remapped = np.full(dimensions, UNKNOWN, dtype=np.uint8)
    for material, output_id in output_ids.items():
        remapped[semantic == source_ids[material]] = output_id
    objects = []
    covered = np.zeros(dimensions, dtype=bool)
    for region in record.get("regions", []):
        material = material_name(region["material"])
        if material not in output_ids:
            continue
        raw = load_label(relative_file(mask_root, region["mask"]))
        if raw.shape != dimensions:
            raise ValueError(record["filename"] + ": region mask/RGB dimensions differ")
        # Exclude areas overwritten by another material (e.g. wheel hubs).
        # Same-class overlapping regions are assigned once within this seed.
        mask = (raw > 0) & (semantic == source_ids[material]) & ~covered
        if mask.any():
            objects.append((material, mask))
            covered |= mask
    if np.any((remapped != UNKNOWN) & ~covered):
        raise ValueError(record["filename"] + ": region masks do not cover known semantic pixels")
    if not objects:
        raise ValueError(record["filename"] + ": no known nonempty region masks")
    return remapped, objects


def save_frame(path, output, labels, confidence, epsilon_by_id, metadata, output_ids, colors):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    known = labels != UNKNOWN
    if labels.shape != rgb.shape[:2]:
        raise ValueError(path.name + ": predictor output/RGB dimensions differ")
    Image.fromarray(labels).save(output / (path.stem + ".png"))
    np.save(output / (path.stem + ".confidence.npy"), confidence.astype(np.float32))
    epsilon = np.full(labels.shape, np.nan, dtype=np.float32)
    overlay = rgb.copy()
    for material, class_id in output_ids.items():
        selected = labels == class_id
        epsilon[selected] = epsilon_by_id[class_id]
        overlay[selected] = np.rint(0.55 * rgb[selected] + 0.45 * np.array(
            colors[material])).astype(np.uint8)
    Image.fromarray(overlay).save(output / "overlay" / (path.stem + ".png"))
    np.save(output / "emissivity" / (path.stem + ".npy"), epsilon)
    return {"filename": path.name, **metadata,
            "known_fraction": float(known.mean()),
            "mask": path.stem + ".png", "confidence": path.stem + ".confidence.npy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True, type=Path,
                        help="Flat directory containing the ordered RGB sequence")
    parser.add_argument("--mask_manifest", required=True, type=Path,
                        help="masks_sample/manifest.json from json_to_mask.py")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="New/empty output directory; label PNGs go directly here")
    parser.add_argument("--checkpoint", type=Path,
                        help="SAM2 checkpoint; required unless --validate_only")
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2_root", type=Path, default=Path(__file__).resolve().parents[1]
                        / "third_party" / "sam2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame_list", type=Path,
                        help="Optional subset: one filename/stem per line; comments start with #")
    parser.add_argument("--logit_threshold", type=float, default=0.0)
    parser.add_argument("--unknown_epsilon0", type=float, default=0.9,
                        help="Explicit fallback assumption for later training, not a measurement")
    parser.add_argument("--tmp_dir", type=Path, help="Optional temporary JPEG parent directory")
    parser.add_argument("--validate_only", action="store_true",
                        help="Validate schema, priors and all keyframe masks; no model or output writes")
    parser.add_argument("--learn_materials", nargs="+", default=[],
                        help="Explicit materials enabling both K and R; default keeps all fixed")
    args = parser.parse_args()
    if not math.isfinite(args.logit_threshold):
        raise ValueError("--logit_threshold must be finite")
    if not 0.01 <= args.unknown_epsilon0 <= 0.99:
        raise ValueError("--unknown_epsilon0 must be in [0.01, 0.99]")
    paths = sorted((path for path in args.image_dir.iterdir() if path.is_file()
                    and path.suffix.lower() in SUFFIXES), key=natural_key)
    if args.frame_list:
        requests = [line.strip() for line in args.frame_list.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        requested = set(requests)
        available = {path.name for path in paths} | {path.stem for path in paths}
        if requested - available:
            raise ValueError("Frames absent from --image_dir: " + str(sorted(requested - available)))
        paths = [path for path in paths if path.name in requested or path.stem in requested]
    if not paths:
        raise ValueError("No selected RGB frames")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Image stems must be unique")
    dimensions = None
    for path in paths:
        with Image.open(path) as image:
            shape = (image.height, image.width)
        if dimensions is not None and dimensions != shape:
            raise ValueError("All frames must have identical original dimensions")
        dimensions = shape
    manifest = read_json(args.mask_manifest)
    mask_root = args.mask_manifest.resolve().parent
    class_document = read_json(mask_root / "class_map.json")
    records = {row["filename"]: row for row in manifest["frames"]}
    if len(records) != len(manifest["frames"]):
        raise ValueError("Duplicate frame records in input manifest")
    source_ids, source_unknown, output_ids, colors, epsilon_by_id = read_material_schema(
        class_document, records)
    materials = material_rows(output_ids, epsilon_by_id, args.learn_materials)
    seeds = []
    for index, path in enumerate(paths):
        record = records.get(path.name)
        if record is not None and any(material_name(row.get("material", "unknown")) in output_ids
                                      for row in record.get("regions", [])):
            labels, objects = load_keyframe(record, mask_root, source_ids,
                                             source_unknown, dimensions, output_ids)
            seeds.append((index, labels, objects))
    if not seeds:
        raise ValueError("No labelled keyframes match selected RGB filenames")
    ignored = sorted(set(records) - {path.name for path in paths})
    if ignored:
        print("[SAM2] Keyframes outside the selected subset are unused: " + ", ".join(ignored))
    if args.validate_only:
        print("[SAM2] Validated: frames=%d, keyframes=%d, materials=%d; no inference/output writes." % (
            len(paths), len(seeds), len(output_ids)))
        for row in materials:
            print("  id=%d name=%s epsilon0=%.6g learn_k=%s learn_delta=%s" % (
                row["id"], row["name"], row["epsilon0"], row["learn_k"], row["learn_delta"]))
        return
    if args.output_dir.exists() and (not args.output_dir.is_dir()
                                     or any(args.output_dir.iterdir())):
        raise ValueError("Output must be new or empty: " + str(args.output_dir))
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: " + str(args.checkpoint))
    if not (args.sam2_root / "sam2" / "build_sam.py").is_file():
        raise FileNotFoundError("Invalid --sam2_root: " + str(args.sam2_root))
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sys.path.insert(0, str(args.sam2_root.resolve()))
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("overlay", "emissivity"):
        (args.output_dir / name).mkdir()
    if args.tmp_dir:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "material_config.json", {
        "format_version": 1, "temperature_reference_K": 300.0,
        "spectral_band": "8-14 um approximation; verify actual sensor",
        "unknown_label": UNKNOWN, "unknown_epsilon0": args.unknown_epsilon0,
        "confirmation_note": "confirmed means manually supplied assumptions, not measured ground truth",
        "materials": materials})
    write_json(args.output_dir / "class_map.json", {
        "format_version": 2, "unknown_label": UNKNOWN, "unknown_id": UNKNOWN,
        "classes": [{"id": class_id, "name": name, "color_rgb": colors[name]}
                    for name, class_id in output_ids.items()] +
                   [{"id": UNKNOWN, "name": "unknown", "color_rgb": [0, 0, 0]}],
        "materials": materials,
        "source_class_ids": source_ids, "source_unknown_id": source_unknown})
    boundaries = [0] + [(seeds[i][0] + seeds[i + 1][0]) // 2 + 1
                        for i in range(len(seeds) - 1)] + [len(paths)]
    reports = []
    for segment, (seed_index, seed_label, objects) in enumerate(seeds):
        begin, end = boundaries[segment], boundaries[segment + 1]
        chunk = paths[begin:end]
        seed_local = seed_index - begin
        print("[SAM2] keyframe=%s, segment=%d..%d, objects=%d" % (
            paths[seed_index].name, begin, end - 1, len(objects)))
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
               else nullcontext())
        with tempfile.TemporaryDirectory(prefix="via_sam2_", dir=args.tmp_dir) as temp_dir:
            temp_root = Path(temp_dir)
            for index, path in enumerate(chunk):
                with Image.open(path) as image:
                    image.convert("RGB").save(temp_root / ("%06d.jpg" % index), quality=100)
            with torch.inference_mode(), amp:
                state = predictor.init_state(str(temp_root), offload_video_to_cpu=True,
                                             offload_state_to_cpu=True)
                object_material = {}
                for object_id, (material, mask) in enumerate(objects, start=1):
                    object_material[object_id] = material
                    predictor.add_new_mask(state, frame_idx=seed_local, obj_id=object_id, mask=mask)
                written = set()
                for reverse in ([False, True] if seed_local > 0 else [False]):
                    for index, object_ids, logits in predictor.propagate_in_video(
                            state, start_frame_idx=seed_local, reverse=reverse):
                        index = int(index)
                        if index in written:
                            continue
                        scores = logits[:, 0].float()
                        best, winners = scores.max(dim=0)
                        known = best > args.logit_threshold
                        label = torch.full(best.shape, UNKNOWN, dtype=torch.uint8, device=best.device)
                        for local, object_id in enumerate(object_ids):
                            label[known & (winners == local)] = output_ids[object_material[int(object_id)]]
                        labels = label.cpu().numpy()
                        confidence = torch.where(known, torch.sigmoid(best), 0.0).cpu().numpy()
                        manual = index == seed_local
                        if manual:
                            # Direct-mask prompting is not automatic edge refinement.
                            # Preserve the human seed exactly, including its unknown areas.
                            labels = seed_label.copy()
                            confidence = (labels != UNKNOWN).astype(np.float32)
                        reports.append(save_frame(chunk[index], args.output_dir, labels,
                                                  confidence, epsilon_by_id, {
                            "source_keyframe": paths[seed_index].name,
                            "manual_keyframe": manual, "segment": segment,
                            "selected_sequence_index": begin + index}, output_ids, colors))
                        written.add(index)
                if len(written) != len(chunk):
                    raise RuntimeError("Incomplete segment coverage: %d/%d" % (len(written), len(chunk)))
                del state
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    if len(reports) != len(paths):
        raise RuntimeError("Output frame count does not match selected inputs")
    reports.sort(key=lambda row: row["selected_sequence_index"])
    write_json(args.output_dir / "propagation_manifest.json", {
        "source_mask_manifest": str(args.mask_manifest.resolve()),
        "image_dir": str(args.image_dir.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "model_cfg": args.model_cfg, "frame_list": str(args.frame_list) if args.frame_list else None,
        "strategy": "nearest-keyframe intervals, per-region local objects, bidirectional propagation",
        "logit_threshold": args.logit_threshold,
        "notes": ["Output labels are RGB-coordinate labels, not registered TIR labels.",
                  "Source class IDs are read from class_map.json; output materials are contiguous from 0, unknown=255.",
                  "K/R learning is disabled unless explicitly enabled with --learn_materials.",
                  "Objects are reset at segment boundaries; check discontinuities in overlays.",
                  "No matching across keyframes is claimed; missing/new parts need manual prompts.",
                  "Confidence is an uncalibrated sigmoid logit; manual known pixels have weight 1.",
                  "SAM2 direct mask prompts do not automatically refine manual seed edges."],
        "frames": reports})
    print("[SAM2] Done: %d frames, %d keyframes -> %s" % (
        len(paths), len(seeds), args.output_dir.resolve()))


if __name__ == "__main__":
    main()
