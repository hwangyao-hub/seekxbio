from __future__ import annotations

from collections import Counter
from pathlib import Path
import ast
import json
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .utils import resolve_device


def render_counts_overlay(image: Image.Image, counts: dict[str, int]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_lines = [f"{k}: {v}" for k, v in counts.items()]
    text = "Counts\n" + "\n".join(text_lines) if text_lines else "Counts\nNone"

    # Simple background box for readability
    padding = 6
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font)
    box_w = text_bbox[2] - text_bbox[0] + padding * 2
    box_h = text_bbox[3] - text_bbox[1] + padding * 2
    draw.rectangle([10, 10, 10 + box_w, 10 + box_h], fill=(0, 0, 0))
    draw.multiline_text((10 + padding, 10 + padding), text, fill=(255, 255, 255), font=font)
    return image


def _result_to_detections(result, names: dict) -> list[dict]:
    dets: list[dict] = []
    if result.boxes is None:
        return dets
    xyxy = result.boxes.xyxy.tolist()
    confs = result.boxes.conf.tolist() if result.boxes.conf is not None else []
    clses = result.boxes.cls.tolist() if result.boxes.cls is not None else []
    for i, box in enumerate(xyxy):
        cls_id = int(clses[i]) if i < len(clses) else -1
        label = names.get(cls_id, str(cls_id))
        conf = float(confs[i]) if i < len(confs) else 0.0
        dets.append(
            {
                "class_id": cls_id,
                "label": label,
                "confidence": conf,
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            }
        )
    return dets


def render_detections(
    image: Image.Image,
    detections: list[dict],
    font_size: int = 22,
    line_width: int = 4,
    label_mapping: dict[int, str] | None = None,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = None
    for font_name in ("msyh.ttc", "msyh.ttf", "simsun.ttc", "simhei.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        if label_mapping and det.get("class_id", -1) >= 0:
            cid = int(det["class_id"])
            if cid in label_mapping:
                label = label_mapping[cid]
        conf = det["confidence"]
        text = f"{label} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(37, 99, 235), width=line_width)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=(37, 99, 235))
        draw.text((x1 + 2, y1 - th - 2), text, fill=(255, 255, 255), font=font)
    return image


def load_class_names(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    data = p.read_bytes()
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            text = data.decode(enc)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                return lines
        except Exception:
            continue
    return []


def load_label_mapping_py(path: str) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    data = p.read_bytes()
    text = ""
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            text = data.decode(enc)
            break
        except Exception:
            continue
    if not text:
        return {}
    try:
        tree = ast.parse(text)
    except Exception:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "Chinese_name":
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return {}
    return {}


def load_class_mapping(yaml_path: str, label_py_path: str) -> dict[int, str]:
    try:
        import yaml
    except Exception:
        return {}
    ypath = Path(yaml_path)
    if not ypath.exists():
        return {}
    data = yaml.safe_load(ypath.read_text(encoding="utf-8"))
    names = data.get("names", [])
    if not isinstance(names, list):
        return {}
    label_map = load_label_mapping_py(label_py_path)
    if not label_map:
        return {}
    mapping: dict[int, str] = {}
    for idx, name in enumerate(names):
        if name in label_map:
            mapping[idx] = label_map[name]
    return mapping


def load_class_mapping_csv(path: str) -> dict[int, str]:
    p = Path(path)
    if not p.exists():
        return {}
    data = p.read_bytes()
    text = ""
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            text = data.decode(enc)
            break
        except Exception:
            continue
    if not text:
        return {}
    mapping: dict[int, str] = {}
    for row in csv.reader(text.splitlines()):
        if len(row) < 2:
            continue
        try:
            idx = int(row[0])
        except Exception:
            continue
        name = row[1].strip()
        if name:
            mapping[idx] = name
    return mapping


def export_xanylabeling_json(
    image_path: str,
    image_size: tuple[int, int],
    detections: list[dict],
    output_json: str,
    class_names: list[str] | None = None,
    label_mapping: dict[int, str] | None = None,
) -> str:
    width, height = image_size
    shapes = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        if label_mapping and det.get("class_id", -1) >= 0:
            cid = int(det["class_id"])
            if cid in label_mapping:
                label = label_mapping[cid]
        if class_names is not None and det.get("class_id", -1) >= 0:
            cid = int(det["class_id"])
            if cid < len(class_names):
                label = class_names[cid]
        shapes.append(
            {
                "label": label,
                "score": det["confidence"],
                "points": [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )

    data = {
        "version": "3.3.5",
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(image_path).name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def infer_and_count(
    weights: str,
    source_image: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "auto",
    return_dets: bool = False,
    label_mapping: dict[int, str] | None = None,
) -> tuple[Image.Image, dict[int, int], int] | tuple[Image.Image, dict[int, int], int, list[dict]]:
    """
    Run YOLOv8 inference on a single image and return:
      - visualization image (with bbox + labels + counts overlay)
      - per-class counts
      - total count
    """
    weights_path = Path(weights)
    source_path = Path(source_image)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(source_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=resolve_device(device),
        verbose=False,
    )

    result = results[0]
    names = model.names

    detections = _result_to_detections(result, names)
    counts = {int(k): v for k, v in sorted(Counter(int(d["class_id"]) for d in detections).items(), key=lambda kv: kv[0])}
    total = sum(counts.values())

    orig = result.orig_img
    if isinstance(orig, np.ndarray) and orig.ndim >= 2:
        vis_rgb = Image.fromarray(orig[:, :, ::-1])
    else:
        vis_rgb = Image.open(source_path).convert("RGB")
    vis_rgb = render_detections(vis_rgb, detections, label_mapping=label_mapping)
    vis_rgb = render_counts_overlay(vis_rgb, {str(k): v for k, v in counts.items()})

    if return_dets:
        return vis_rgb, counts, total, detections
    return vis_rgb, counts, total
