"""Propagate human-labelled material instances through an ordered RGB sequence.

Each prompted object needs a stable integer ``object_id`` and a human-provided
``material`` name. Prompts may be placed on several keyframes to correct drift.
SAM2 propagates object regions; it never chooses the material name.
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--material_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2_root", default=str(Path(__file__).resolve().parents[1] / "third_party" / "sam2"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SAM2 CUDA requested but CUDA is unavailable")
    sys.path.insert(0, str(Path(args.sam2_root).resolve()))
    from sam2.build_sam import build_sam2_video_predictor

    with open(args.material_config, encoding="utf-8") as handle:
        material_document = json.load(handle)
    material_ids = {str(row["name"]): int(row["id"]) for row in material_document["materials"]}
    unknown_label = int(material_document.get("unknown_label", 255))
    if not material_ids or sorted(material_ids.values()) != list(range(len(material_ids))):
        raise ValueError("Material ids must be unique contiguous integers starting at zero")
    if not 0 <= unknown_label <= 255 or unknown_label in material_ids.values():
        raise ValueError("unknown_label must be an unused 8-bit value")
    with open(args.prompts, encoding="utf-8") as handle:
        prompt_document = json.load(handle)
    prompts = prompt_document.get("frames", prompt_document)
    image_root = Path(args.image_dir)
    def natural_key(path):
        return tuple((1, int(part)) if part.isdigit() else (0, part.lower())
                     for part in re.split(r"(\d+)", path.name))
    image_paths = sorted((path for path in image_root.iterdir()
                          if path.is_file() and path.suffix.lower() in
                          {".png", ".jpg", ".jpeg", ".tif", ".tiff"}), key=natural_key)
    if not image_paths:
        raise RuntimeError("No input frames found")
    frame_index = {path.stem: idx for idx, path in enumerate(image_paths)}
    if len(frame_index) != len(image_paths):
        raise ValueError("Input image stems must be unique")
    unknown_frames = sorted(set(prompts) - set(frame_index))
    if unknown_frames:
        raise ValueError(f"Prompts reference absent frames: {unknown_frames[:8]}")
    object_material = {}

    with tempfile.TemporaryDirectory(prefix="sam2_material_frames_") as temp_dir:
        temp_root = Path(temp_dir)
        for idx, path in enumerate(image_paths):
            Image.open(path).convert("RGB").save(temp_root / f"{idx:06d}.jpg", quality=100)
        predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=args.device)
        state = predictor.init_state(video_path=str(temp_root), offload_video_to_cpu=True)
        with torch.inference_mode():
            for stem, annotation in prompts.items():
                idx = frame_index[stem]
                for object_prompt in annotation.get("objects", []):
                    object_id = int(object_prompt["object_id"])
                    material = str(object_prompt["material"])
                    if material not in material_ids:
                        raise ValueError(f"Unknown material {material!r} in frame {stem}")
                    previous = object_material.setdefault(object_id, material)
                    if previous != material:
                        raise ValueError(f"object_id {object_id} changes material from {previous} to {material}")
                    positive = object_prompt.get("positive_points", [])
                    negative = object_prompt.get("negative_points", [])
                    points = np.asarray(positive + negative, dtype=np.float32) if positive or negative else None
                    labels = (np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32)
                              if points is not None else None)
                    box = object_prompt.get("box")
                    box = np.asarray(box, dtype=np.float32) if box is not None else None
                    if points is None and box is None:
                        raise ValueError(f"Object {object_id} in frame {stem} has no prompt")
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=idx, obj_id=object_id,
                        points=points, labels=labels, box=box)

            output_root = Path(args.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            if not object_material:
                raise ValueError("No object prompts were provided")
            written = set()
            first_prompt_idx = min(frame_index[stem] for stem, annotation in prompts.items()
                                   if annotation.get("objects"))
            directions = [False] + ([True] if first_prompt_idx > 0 else [])
            for reverse in directions:
                for idx, object_ids, mask_logits in predictor.propagate_in_video(
                        state, start_frame_idx=first_prompt_idx, reverse=reverse):
                    if int(idx) in written:
                        continue
                    logits = mask_logits[:, 0].float()
                    best_logits, best_index = logits.max(dim=0)
                    confidence = torch.where(best_logits > 0, torch.sigmoid(best_logits), 0.0)
                    label = torch.full(best_logits.shape, unknown_label, dtype=torch.uint8, device=best_logits.device)
                    for local_index, object_id in enumerate(object_ids):
                        selected = (best_index == local_index) & (best_logits > 0.0)
                        label[selected] = material_ids[object_material[int(object_id)]]
                    stem = image_paths[int(idx)].stem
                    Image.fromarray(label.cpu().numpy(), mode="L").save(output_root / f"{stem}.png")
                    np.save(output_root / f"{stem}.confidence.npy", confidence.cpu().numpy().astype(np.float32))
                    written.add(int(idx))
            if len(written) != len(image_paths):
                raise RuntimeError(f"SAM2 propagated {len(written)}/{len(image_paths)} frames")
    print(f"[SAM2] propagated {len(object_material)} labelled objects over {len(image_paths)} frames")


if __name__ == "__main__":
    main()
