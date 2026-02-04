"""
YOLOv8 training script for microscopy cell detection & morphology classification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.train_core import train_yolov8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on microscopy cell dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda or cuda:0.")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="cells")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0, help="Use first N train images (0=all).")
    parser.add_argument("--limit-val", type=int, default=0, help="Use first N val images (0=all).")
    parser.add_argument(
        "--remap-classes",
        action="store_true",
        help="Remap classes to remove gaps by rewriting temp labels.",
    )
    parser.add_argument(
        "--no-freeze-splits",
        action="store_false",
        dest="freeze_splits",
        help="Disable freezing train/val lists into a snapshot YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_yolov8(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        resume=args.resume,
        limit_train_images=args.limit_train,
        limit_val_images=args.limit_val,
        remap_classes=args.remap_classes,
        freeze_splits=args.freeze_splits,
    )


if __name__ == "__main__":
    main()
