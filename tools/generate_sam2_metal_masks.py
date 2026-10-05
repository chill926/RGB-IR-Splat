"""Generate multi-class material masks from human-labelled SAM2 prompts.

The filename is retained for command compatibility, but the output is no
longer binary metal/non-metal. Material ids and fixed epsilon_0 values come
from --material_config. Prompt JSON format:

{
  "frames": {
    "000001": {"objects": [
      {"material": "paint", "box": [x0,y0,x1,y1],
       "positive_points": [[x,y]], "negative_points": [[x,y]]}
    ]}
  }
}

Pixels not covered by a confirmed object remain unknown (normally label 255).
SAM2 segments prompted regions; it does not infer material identity.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--material_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2_root", default=str(Path(__file__).resolve().parents[1] / "third_party" / "sam2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_unprompted", action="store_true", default=False,
                        help="Allow frames with no prompts; their complete mask is unknown")
    return parser.parse_args()


def object_prompt(annotation):
    positive = annotation.get("positive_points", [])
    negative = annotation.get("negative_points", [])
    points = np.asarray(positive + negative, dtype=np.float32) if positive or negative else None
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32) if points is not None else None
    box = annotation.get("box")
    box = np.asarray(box, dtype=np.float32) if box is not None else None
    if points is None and box is None:
        raise ValueError("Each material object needs a box and/or point prompts")
    return points, labels, box


def main():
    args = parse_args()
    sam2_root = Path(args.sam2_root).resolve()
    sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SAM2 CUDA requested but CUDA is unavailable")
    with open(args.material_config, encoding="utf-8") as handle:
        material_document = json.load(handle)
    material_ids = {str(row["name"]): int(row["id"]) for row in material_document.get("materials", [])}
    if sorted(material_ids.values()) != list(range(len(material_ids))):
        raise ValueError("Material ids must be contiguous integers starting at zero")
    unknown_label = int(material_document.get("unknown_label", 255))
    if not 0 <= unknown_label <= 255 or unknown_label in material_ids.values():
        raise ValueError("unknown_label must be an unused 8-bit value")
    with open(args.prompts, encoding="utf-8") as handle:
        prompt_document = json.load(handle)
    prompts = prompt_document.get("frames", prompt_document)

    model = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(model)
    image_root, output_root = Path(args.image_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in image_root.iterdir()
                         if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    missing_prompts = [path.stem for path in image_paths if path.stem not in prompts]
    if missing_prompts and not args.allow_unprompted:
        preview = ", ".join(missing_prompts[:8])
        raise RuntimeError(
            f"Prompt JSON is missing {len(missing_prompts)} images ({preview}). "
            "Use an empty objects list when every pixel should remain unknown."
        )

    for image_path in image_paths:
        annotation = prompts.get(image_path.stem, {"objects": []})
        image = np.asarray(Image.open(image_path).convert("RGB"))
        predictor.set_image(image)
        label_image = np.full(image.shape[:2], unknown_label, dtype=np.uint8)
        confidence = np.zeros(image.shape[:2], dtype=np.float32)
        for material_object in annotation.get("objects", []):
            name = str(material_object.get("material", ""))
            if name not in material_ids:
                raise ValueError(f"Unknown material {name!r} in frame {image_path.stem}")
            points, labels, box = object_prompt(material_object)
            masks, scores, _ = predictor.predict(
                point_coords=points, point_labels=labels, box=box, multimask_output=True)
            choice = int(np.argmax(scores))
            mask, score = masks[choice].astype(bool), float(scores[choice])
            update = mask & (score >= confidence)
            label_image[update] = material_ids[name]
            confidence[update] = score
        Image.fromarray(label_image, mode="L").save(output_root / f"{image_path.stem}.png")
        np.save(output_root / f"{image_path.stem}.confidence.npy", confidence)
        counts = {name: int((label_image == material_id).sum())
                  for name, material_id in material_ids.items()}
        print(f"[SAM2] {image_path.name}: materials={counts}, unknown={int((label_image == unknown_label).sum())}")


if __name__ == "__main__":
    main()
