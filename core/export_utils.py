"""
多格式导出工具
支持: COCO JSON, Pascal VOC XML, X-AnyLabeling JSON
"""
from __future__ import annotations

from pathlib import Path
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime


def export_coco_json(
    image_paths: list[str],
    all_detections: dict[str, list[dict]],
    output_path: str,
    class_names: dict[int, str] | None = None,
) -> str:
    """
    导出为 COCO JSON 格式。

    Args:
        image_paths: 图片路径列表
        all_detections: {image_name: [detection_dict, ...]}
        output_path: 输出JSON路径
        class_names: 类别ID到名称的映射
    """
    from PIL import Image

    coco = {
        "info": {
            "description": "MicroscopyAI Export",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # 构建类别列表
    all_class_ids = set()
    for dets in all_detections.values():
        for det in dets:
            all_class_ids.add(det.get("class_id", 0))

    for cls_id in sorted(all_class_ids):
        name = class_names.get(cls_id, f"class_{cls_id}") if class_names else f"class_{cls_id}"
        coco["categories"].append(
            {
                "id": cls_id,
                "name": name,
                "supercategory": "cell",
            }
        )

    ann_id = 1
    for img_id, img_path in enumerate(image_paths, 1):
        img = Image.open(img_path)
        w, h = img.size

        coco["images"].append(
            {
                "id": img_id,
                "file_name": Path(img_path).name,
                "width": w,
                "height": h,
            }
        )

        img_name = Path(img_path).name
        if img_name in all_detections:
            for det in all_detections[img_name]:
                x1, y1, x2, y2 = det["bbox"]
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": det.get("class_id", 0),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                        "score": det.get("confidence", 1.0),
                    }
                )
                ann_id += 1

    Path(output_path).write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def export_pascal_voc_xml(
    image_path: str,
    detections: list[dict],
    output_path: str,
    class_names: dict[int, str] | None = None,
) -> str:
    """
    导出为 Pascal VOC XML 格式。
    """
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size

    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = str(Path(image_path).parent.name)
    ET.SubElement(root, "filename").text = Path(image_path).name

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"

    for det in detections:
        obj = ET.SubElement(root, "object")

        cls_id = det.get("class_id", 0)
        name = class_names.get(cls_id, f"class_{cls_id}") if class_names else det.get(
            "label", f"class_{cls_id}"
        )
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        if "confidence" in det:
            ET.SubElement(obj, "confidence").text = f"{det['confidence']:.4f}"

        bndbox = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = det["bbox"]
        ET.SubElement(bndbox, "xmin").text = str(int(x1))
        ET.SubElement(bndbox, "ymin").text = str(int(y1))
        ET.SubElement(bndbox, "xmax").text = str(int(x2))
        ET.SubElement(bndbox, "ymax").text = str(int(y2))

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    Path(output_path).write_text(xml_str, encoding="utf-8")
    return output_path
