"""
YOLOv8 inference + per-class counting + visualization for microscopy images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from core.infer_core import infer_and_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer and count cells in a single microscopy image.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights.")
    parser.add_argument("--source", type=str, required=True, help="Input image path (PNG/TIFF).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda or cuda:0.")
    parser.add_argument("--save_dir", type=str, default="outputs/infer")
    parser.add_argument(
        "--use_cn_labels",
        action="store_true",
        help="Render labels with built-in Chinese class names if available.",
    )
    parser.add_argument(
        "--save_dets",
        type=str,
        default="",
        help="Optional path to save detections JSON (list of boxes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    source_path = Path(args.source)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    label_mapping = None
    if args.use_cn_labels:
        try:
            from core.constants import CLASS_NAMES_CN

            label_mapping = CLASS_NAMES_CN
        except Exception:
            label_mapping = None

    vis_rgb, counts, _, dets = infer_and_count(
        weights=str(weights_path),
        source_image=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        return_dets=True,
        label_mapping=label_mapping,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_image = save_dir / f"{source_path.stem}_pred.png"
    out_json = save_dir / f"{source_path.stem}_counts.json"
    dets_json = Path(args.save_dets) if args.save_dets else save_dir / f"{source_path.stem}_dets.json"

    vis_rgb.save(out_image)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in counts.items()}, f, ensure_ascii=False, indent=2)
    with dets_json.open("w", encoding="utf-8") as f:
        json.dump(dets, f, ensure_ascii=False, indent=2)

    print(f"Image: {source_path}")
    print(f"Saved visualization: {out_image}")
    print(f"Saved counts: {out_json}")
    print("Counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
