from __future__ import annotations

CLASS_NAMES_CN = {
    0: "嗜碱细胞",
    1: "刺状红细胞",
    2: "椭圆形红细胞",
    3: "嗜酸细胞",
    4: "红细胞前体",
    5: "低色素症",
    6: "淋巴细胞",
    7: "大细胞",
    8: "小细胞",
    9: "单核细胞",
    10: "中性粒细胞",
    11: "椭圆细胞",
    12: "血小板",
    13: "红细胞-猫",
    14: "红细胞-狗",
    15: "裂片细胞",
    16: "球形细胞",
    17: "口形细胞",
    18: "靶细胞",
    19: "泪滴细胞",
    20: "白细胞",
}

MODEL_OPTIONS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolov5n.pt",
    "yolov5s.pt",
    "yolov5m.pt",
    "yolov5l.pt",
    "yolov5x.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
]

DEVICE_OPTIONS = ["auto", "cpu", "cuda", "cuda:0", "cuda:1"]
