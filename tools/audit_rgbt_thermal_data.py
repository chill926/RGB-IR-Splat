"""Audit RGBT-Scenes thermal files before selecting an observation domain."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image


def inspect_image(path):
    with Image.open(path) as image:
        array = np.asarray(image)
        exif = image.getexif()
        metadata = {ExifTags.TAGS.get(key, str(key)): str(value) for key, value in exif.items()}
        channels_equal = None
        if array.ndim == 3 and array.shape[2] >= 3:
            channels_equal = bool(np.array_equal(array[..., 0], array[..., 1]) and
                                  np.array_equal(array[..., 0], array[..., 2]))
        return {
            "path": str(path), "format": image.format, "mode": image.mode,
            "dtype": str(array.dtype), "shape": list(array.shape),
            "min": float(np.min(array)), "max": float(np.max(array)),
            "channels_equal": channels_equal, "exif": metadata,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scene = Path(args.scene)
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rows, missing_pairs = [], []
    for split in ("train", "test"):
        thermal_root, rgb_root = scene / "thermal" / split, scene / "rgb" / split
        for path in sorted(item for item in thermal_root.iterdir() if item.suffix.lower() in extensions):
            rows.append({"split": split, **inspect_image(path)})
            if not any((rgb_root / f"{path.stem}{suffix}").exists() for suffix in extensions):
                missing_pairs.append(str(path))
    signatures = sorted({(row["format"], row["mode"], row["dtype"], tuple(row["shape"][2:]))
                         for row in rows}, key=str)
    report = {
        "scene": str(scene.resolve()), "image_count": len(rows),
        "signatures": [list(signature) for signature in signatures],
        "missing_rgb_pairs": missing_pairs, "images": rows,
        "interpretation": (
            "This audit cannot prove radiometric calibration. Use apparent_temperature only when "
            "the dataset documentation or metadata supplies a reversible raw-to-temperature mapping; "
            "otherwise retain normalized_dn."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({key: report[key] for key in ("scene", "image_count", "signatures", "missing_rgb_pairs")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
