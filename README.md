# Microscopy YOLOv8 Cell Detection, Classification & Counting

This project provides a reproducible YOLOv8 workflow for microscope cell detection, morphology classification, and per-class counting.

## 1) Project structure

```
microscopy-yolov8/
  data/
    cell.yaml                  # YOLO dataset config (edit classes/paths)
  docs/
    dataset_format.md          # Dataset + label format description
  outputs/
    infer/                     # Inference visualizations + counts
  scripts/
    train.py                   # Training entry point
    infer_and_count.py         # Inference + counting + visualization
```

## 2) Quick start

Install dependencies (example):

```
pip install ultralytics torch torchvision pillow gradio
```

Train:

```
python scripts/train.py --data data/cell.yaml --model yolov8n.pt --epochs 100 --imgsz 640
```

Inference + counting:

```
python scripts/infer_and_count.py --weights runs/detect/train/weights/best.pt --source path/to/image.png --imgsz 640 --conf 0.25
```

Launch UI:

```
python app.py
```

## 3) Notes for microscopy

- If magnification varies (20x/25x), use data augmentation and ensure training set includes both.
- For dense or mildly overlapping cells, start with a smaller model (e.g., yolov8n/s) and tune NMS IoU.
- This pipeline uses detection only (no segmentation).

See `docs/dataset_format.md` for dataset and label format.
