# Dataset & Label Format (YOLO)

## Directory layout

```
datasets/cells/
  images/
    train/
      *.png | *.tif | *.tiff
    val/
      *.png | *.tif | *.tiff
  labels/
    train/
      *.txt
    val/
      *.txt
```

Each image has a label file with the same stem, e.g.:

```
images/train/field_001.png
labels/train/field_001.txt
```

## Label file format

Each line is one object:

```
class_id x_center y_center width height
```

All coordinates are **normalized** to `[0,1]` relative to image width/height.

Example:

```
0 0.512 0.438 0.083 0.072
2 0.214 0.665 0.051 0.049
```

## Class list

Define class names in `data/cell.yaml` under `names`.

## TIFF notes

Ultralytics supports common image types. If multi-page TIFFs are used, convert to single-page images first.
