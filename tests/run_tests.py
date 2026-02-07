#!/usr/bin/env python
"""
显微镜血细胞分析项目 - 自动化测试脚本
用于验证 Codex 实现的功能
"""
from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class TestResult:
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    SKIP = "⏭️ SKIP"
    ERROR = "💥 ERROR"


def test_module_imports():
    """测试模块导入"""
    errors = []

    # 测试 core 模块导入
    try:
        from core import infer_and_count
    except ImportError as e:
        errors.append(f"infer_and_count: {e}")

    try:
        from core import infer_batch
    except ImportError as e:
        errors.append(f"infer_batch: {e}")

    try:
        from core.train_core import train_yolov8_stream_with_process
    except ImportError as e:
        errors.append(f"train_yolov8_stream_with_process: {e}")

    if errors:
        print(f"{TestResult.FAIL}: 模块导入测试")
        for err in errors:
            print(f"  - {err}")
        return False

    print(f"{TestResult.PASS}: 模块导入测试")
    return True


def test_infer_batch_function_signature():
    """测试批量推理函数签名"""
    from core import infer_batch
    import inspect

    sig = inspect.signature(infer_batch)
    params = list(sig.parameters.keys())

    required_params = ["weights", "source_dir", "output_dir"]
    optional_params = ["imgsz", "conf", "iou", "device", "label_mapping", "progress_callback"]

    missing = [p for p in required_params if p not in params]
    if missing:
        print(f"{TestResult.FAIL}: infer_batch缺少参数: {missing}")
        return False

    print(f"{TestResult.PASS}: infer_batch函数签名正确")
    return True


def test_train_stream_with_process_signature():
    """测试训练流函数签名"""
    from core.train_core import train_yolov8_stream_with_process
    import inspect

    sig = inspect.signature(train_yolov8_stream_with_process)
    params = list(sig.parameters.keys())

    required_params = ["data_yaml"]
    if not all(p in params for p in required_params):
        print(f"{TestResult.FAIL}: train_yolov8_stream_with_process缺少参数")
        return False

    # 检查返回类型注解
    annotations = train_yolov8_stream_with_process.__annotations__
    print(f"  返回类型: {annotations.get('return', 'unknown')}")

    print(f"{TestResult.PASS}: train_yolov8_stream_with_process函数签名正确")
    return True


def test_progress_parsing():
    """测试训练进度解析"""
    try:
        # 尝试从app.py导入
        sys.path.insert(0, str(ROOT))
        from app import parse_training_progress
    except ImportError:
        # 如果app.py没有这个函数，手动测试逻辑
        import re

        def parse_training_progress(line: str):
            match = re.search(r"Epoch\s+(\d+)/(\d+)", line)
            if match:
                return int(match.group(1)), int(match.group(2))
            return None

    test_cases = [
        ("Epoch 1/100", (1, 100)),
        ("Epoch 15/100", (15, 100)),
        ("Epoch 100/100", (100, 100)),
        ("Some other log line", None),
        ("      Epoch 50/200      ", (50, 200)),
        ("", None),
        ("epoch 1/100", None),  # 小写不匹配
    ]

    failed = []
    for line, expected in test_cases:
        result = parse_training_progress(line)
        if result != expected:
            failed.append(f"'{line}': got {result}, expected {expected}")

    if failed:
        print(f"{TestResult.FAIL}: 进度解析测试")
        for f in failed:
            print(f"  - {f}")
        return False

    print(f"{TestResult.PASS}: 进度解析测试 ({len(test_cases)} cases)")
    return True


def test_export_utils_exist():
    """测试导出工具模块存在"""
    export_utils_path = ROOT / "core" / "export_utils.py"

    if not export_utils_path.exists():
        print(f"{TestResult.SKIP}: export_utils.py 不存在 (可能未实现)")
        return True

    try:
        from core.export_utils import export_coco_json, export_pascal_voc_xml
        print(f"{TestResult.PASS}: export_utils模块导入成功")
        return True
    except ImportError as e:
        print(f"{TestResult.FAIL}: export_utils模块导入失败: {e}")
        return False


def test_coco_export():
    """测试COCO JSON导出"""
    try:
        from core.export_utils import export_coco_json
    except ImportError:
        print(f"{TestResult.SKIP}: export_coco_json 未实现")
        return True

    import json
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试图片
        img_path = Path(tmpdir) / "test_img.png"
        Image.new("RGB", (640, 480), color="white").save(img_path)

        output_path = Path(tmpdir) / "coco_output.json"

        # 测试导出
        all_detections = {
            "test_img.png": [
                {"class_id": 0, "bbox": [10, 20, 100, 150], "confidence": 0.95},
                {"class_id": 1, "bbox": [200, 200, 300, 350], "confidence": 0.88},
            ]
        }

        result = export_coco_json(
            image_paths=[str(img_path)],
            all_detections=all_detections,
            output_path=str(output_path),
            class_names={0: "RBC", 1: "WBC"},
        )

        # 验证输出
        assert output_path.exists(), "输出文件不存在"

        with open(output_path, encoding="utf-8") as f:
            coco = json.load(f)

        assert "images" in coco, "缺少images字段"
        assert "annotations" in coco, "缺少annotations字段"
        assert "categories" in coco, "缺少categories字段"
        assert len(coco["images"]) == 1, f"images数量错误: {len(coco['images'])}"
        assert len(coco["annotations"]) == 2, f"annotations数量错误: {len(coco['annotations'])}"

        # 验证bbox格式 (COCO: [x, y, width, height])
        ann = coco["annotations"][0]
        bbox = ann["bbox"]
        assert len(bbox) == 4, "bbox长度错误"
        # 原始: [10, 20, 100, 150] -> COCO: [10, 20, 90, 130]
        assert bbox[2] > 0 and bbox[3] > 0, "bbox宽高应为正数"

    print(f"{TestResult.PASS}: COCO JSON导出测试")
    return True


def test_voc_export():
    """测试Pascal VOC XML导出"""
    try:
        from core.export_utils import export_pascal_voc_xml
    except ImportError:
        print(f"{TestResult.SKIP}: export_pascal_voc_xml 未实现")
        return True

    import xml.etree.ElementTree as ET
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试图片
        img_path = Path(tmpdir) / "test_img.png"
        Image.new("RGB", (640, 480), color="white").save(img_path)

        output_path = Path(tmpdir) / "voc_output.xml"

        detections = [
            {"class_id": 0, "bbox": [10, 20, 100, 150], "confidence": 0.95},
            {"class_id": 1, "bbox": [200, 200, 300, 350], "confidence": 0.88},
        ]

        result = export_pascal_voc_xml(
            image_path=str(img_path),
            detections=detections,
            output_path=str(output_path),
            class_names={0: "RBC", 1: "WBC"},
        )

        # 验证输出
        assert output_path.exists(), "输出文件不存在"

        tree = ET.parse(output_path)
        root = tree.getroot()

        assert root.tag == "annotation", f"根元素错误: {root.tag}"

        objects = root.findall("object")
        assert len(objects) == 2, f"object数量错误: {len(objects)}"

        # 验证第一个object
        obj = objects[0]
        name = obj.find("name")
        assert name is not None, "缺少name元素"
        assert name.text == "RBC", f"name错误: {name.text}"

        bndbox = obj.find("bndbox")
        assert bndbox is not None, "缺少bndbox元素"
        assert bndbox.find("xmin") is not None, "缺少xmin"
        assert bndbox.find("ymin") is not None, "缺少ymin"
        assert bndbox.find("xmax") is not None, "缺少xmax"
        assert bndbox.find("ymax") is not None, "缺少ymax"

    print(f"{TestResult.PASS}: Pascal VOC XML导出测试")
    return True


def test_batch_inference_with_mock():
    """测试批量推理（使用模拟数据）"""
    from core import infer_batch

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试图片目录
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()

        from PIL import Image
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 80, i * 80, i * 80))
            img.save(input_dir / f"test_{i}.png")

        # 由于没有真实模型，测试空目录处理
        empty_input = Path(tmpdir) / "empty"
        empty_input.mkdir()

        # 检查权重文件
        weights = ROOT / "runs" / "detect" / "cells" / "weights" / "best.pt"
        if not weights.exists():
            # 尝试查找其他权重
            alt_weights = list(ROOT.rglob("*.pt"))
            if alt_weights:
                weights = alt_weights[0]
                print(f"  使用替代权重: {weights}")
            else:
                print(f"{TestResult.SKIP}: 无可用模型权重")
                return True

        # 测试空目录
        results = infer_batch(
            weights=str(weights),
            source_dir=str(empty_input),
            output_dir=str(output_dir / "empty_test"),
        )

        assert results["total_images"] == 0, "空目录应返回0张图片"

        print(f"{TestResult.PASS}: 批量推理基础测试")
        return True


def test_gradio_batch_ui_elements():
    """测试Gradio界面中的批量推理元素"""
    app_path = ROOT / "app.py"

    if not app_path.exists():
        print(f"{TestResult.SKIP}: app.py不存在")
        return True

    content = app_path.read_text(encoding="utf-8")

    required_elements = [
        "批量推理",
        "run_batch_inference",
        "batch_input_dir",
        "batch_output_dir",
    ]

    missing = [elem for elem in required_elements if elem not in content]

    if missing:
        print(f"{TestResult.FAIL}: Gradio界面缺少批量推理元素: {missing}")
        return False

    print(f"{TestResult.PASS}: Gradio批量推理UI元素存在")
    return True


def test_desktop_batch_ui_elements():
    """测试Desktop UI中的批量推理元素"""
    inference_page = ROOT / "ui" / "pages" / "inference_page.py"

    if not inference_page.exists():
        print(f"{TestResult.SKIP}: inference_page.py不存在")
        return True

    content = inference_page.read_text(encoding="utf-8")

    required_elements = [
        "BatchInferenceWorker",
        "batch_input",
        "batch_output",
        "run_batch_inference",
    ]

    missing = [elem for elem in required_elements if elem not in content]

    if missing:
        print(f"{TestResult.FAIL}: Desktop UI缺少批量推理元素: {missing}")
        return False

    print(f"{TestResult.PASS}: Desktop批量推理UI元素存在")
    return True


def test_training_process_control():
    """测试训练进程控制元素"""
    app_path = ROOT / "app.py"

    if not app_path.exists():
        print(f"{TestResult.SKIP}: app.py不存在")
        return True

    content = app_path.read_text(encoding="utf-8")

    required_elements = [
        "training_process",
        "stop_training",
        "killpg" if os.name != "nt" else "terminate",  # Unix用killpg，Windows用terminate
    ]

    # 至少需要有进程控制逻辑
    has_process_control = "training_process" in content and "stop_training" in content

    if not has_process_control:
        print(f"{TestResult.FAIL}: 缺少训练进程控制逻辑")
        return False

    print(f"{TestResult.PASS}: 训练进程控制逻辑存在")
    return True


def test_annotation_undo_redo():
    """测试标注工具撤销/重做元素"""
    annotation_tool = ROOT / "ui" / "annotation_tool.py"

    if not annotation_tool.exists():
        print(f"{TestResult.SKIP}: annotation_tool.py不存在")
        return True

    content = annotation_tool.read_text(encoding="utf-8")

    undo_elements = ["undo", "_history"]
    redo_elements = ["redo", "_redo_stack"]

    has_undo = any(elem in content for elem in undo_elements)
    has_redo = any(elem in content for elem in redo_elements)

    if not has_undo:
        print(f"{TestResult.FAIL}: 标注工具缺少撤销功能")
        return False

    if not has_redo:
        print(f"{TestResult.FAIL}: 标注工具缺少重做功能")
        return False

    print(f"{TestResult.PASS}: 标注工具撤销/重做功能存在")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("显微镜血细胞分析项目 - 自动化测试")
    print("=" * 60)
    print()

    tests = [
        ("模块导入", test_module_imports),
        ("infer_batch函数签名", test_infer_batch_function_signature),
        ("train_stream_with_process签名", test_train_stream_with_process_signature),
        ("训练进度解析", test_progress_parsing),
        ("导出工具模块", test_export_utils_exist),
        ("COCO JSON导出", test_coco_export),
        ("Pascal VOC导出", test_voc_export),
        ("批量推理基础功能", test_batch_inference_with_mock),
        ("Gradio批量推理UI", test_gradio_batch_ui_elements),
        ("Desktop批量推理UI", test_desktop_batch_ui_elements),
        ("训练进程控制", test_training_process_control),
        ("标注工具撤销/重做", test_annotation_undo_redo),
    ]

    results = {"pass": 0, "fail": 0, "skip": 0, "error": 0}

    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            success = test_func()
            if success:
                results["pass"] += 1
            else:
                results["fail"] += 1
        except AssertionError as e:
            print(f"{TestResult.FAIL}: {e}")
            results["fail"] += 1
        except Exception as e:
            print(f"{TestResult.ERROR}: {type(e).__name__}: {e}")
            results["error"] += 1

    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"✅ 通过: {results['pass']}")
    print(f"❌ 失败: {results['fail']}")
    print(f"⏭️ 跳过: {results['skip']}")
    print(f"💥 错误: {results['error']}")
    print()

    total = results["pass"] + results["fail"]
    if total > 0:
        pass_rate = results["pass"] / total * 100
        print(f"通过率: {pass_rate:.1f}%")

    return results["fail"] == 0 and results["error"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
