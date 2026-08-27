"""Generate binary metal/non-metal masks with official SAM2 and metal prompts.

The only material labels used by the training pipeline are metal and non-metal.
The JSON prompt file marks metal foreground objects; every other pixel is
non-metal. SAM2 then produces the binary mask (metal=1, non-metal=0):

{"frame_stem": {"objects": [{"box": [x0,y0,x1,y1],
  "positive_points": [[x,y]], "negative_points": [[x,y]]}]}}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2_root", default=str(Path(__file__).resolve().parents[1] / "third_party" / "sam2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_unprompted", action="store_true", default=False,
                        help="Smoke tests only; formal runs require an entry for every image")
    return parser.parse_args()


def object_prompt(annotation):
    positive = annotation.get("positive_points", [])
    negative = annotation.get("negative_points", [])
    points = np.asarray(positive + negative, dtype=np.float32) if positive or negative else None
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32) if points is not None else None
    box = annotation.get("box")
    box = np.asarray(box, dtype=np.float32) if box is not None else None
    if points is None and box is None:
        raise ValueError("Each metal object needs a box and/or point prompts")
    return points, labels, box


def main():
    args = parse_args()
    sam2_root = Path(args.sam2_root).resolve()
    sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SAM2 CUDA requested but CUDA is unavailable")
    with open(args.prompts, encoding="utf-8") as handle:
        prompts = json.load(handle)
    model = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(model)
    image_root, output_root = Path(args.image_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in image_root.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    missing_prompts = [path.stem for path in image_paths if path.stem not in prompts]
    if missing_prompts and not args.allow_unprompted:
        preview = ", ".join(missing_prompts[:8])
        raise RuntimeError(
            f"Prompt JSON is missing {len(missing_prompts)} images ({preview}). "
            "Use an empty objects list for a view containing no metal."
        )
    for image_path in image_paths:
        annotation = prompts.get(image_path.stem)
        if annotation is None:
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"))
        predictor.set_image(image)
        combined = np.zeros(image.shape[:2], dtype=bool)
        for metal_object in annotation.get("objects", []):
            points, labels, box = object_prompt(metal_object)
            masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, box=box,
                                                  multimask_output=True)
            combined |= masks[int(np.argmax(scores))].astype(bool)
        Image.fromarray((combined.astype(np.uint8) * 255), mode="L").save(output_root / f"{image_path.stem}.png")
        print(f"[SAM2] {image_path.name}: metal_pixels={int(combined.sum())}")


if __name__ == "__main__":
    main()
