# 显微镜血细胞分析项目 - 测试方案

## 测试环境准备

### 前置条件
1. 准备测试图片目录: `tests/test_images/` (至少5张显微镜图片)
2. 准备训练好的模型权重: `runs/detect/cells/weights/best.pt`
3. 确保依赖已安装: `pip install -r requirements.txt`

---

## 测试用例

### 一、批量推理功能测试

#### TC-1.1 基础批量推理
**测试目标**: 验证 `infer_batch()` 函数能正确处理目录中的图片

**测试步骤**:
```python
from core import infer_batch

results = infer_batch(
    weights="runs/detect/cells/weights/best.pt",
    source_dir="tests/test_images",
    output_dir="tests/output/batch_test",
    imgsz=640,
    conf=0.25,
    iou=0.45,
    device="auto",
)

print(f"处理图片数: {results['total_images']}")
print(f"检测细胞数: {results['total_cells']}")
print(f"CSV路径: {results['csv_path']}")
```

**预期结果**:
- [ ] `results['total_images']` 等于输入目录中的图片数量
- [ ] `results['total_cells']` >= 0
- [ ] `tests/output/batch_test/` 目录下有对应的 `*_pred.jpg` 和 `*.json` 文件
- [ ] `summary_report.csv` 文件存在且格式正确

#### TC-1.2 进度回调测试
**测试目标**: 验证进度回调函数正常工作

**测试步骤**:
```python
progress_log = []

def progress_cb(current, total, filename):
    progress_log.append((current, total, filename))
    print(f"Progress: {current}/{total} - {filename}")

results = infer_batch(
    weights="runs/detect/cells/weights/best.pt",
    source_dir="tests/test_images",
    output_dir="tests/output/batch_progress",
    progress_callback=progress_cb,
)

print(f"回调次数: {len(progress_log)}")
```

**预期结果**:
- [ ] 回调次数等于图片数量
- [ ] `current` 从 1 递增到 `total`
- [ ] `filename` 为实际文件名

#### TC-1.3 空目录处理
**测试步骤**:
```python
import os
os.makedirs("tests/empty_dir", exist_ok=True)

results = infer_batch(
    weights="runs/detect/cells/weights/best.pt",
    source_dir="tests/empty_dir",
    output_dir="tests/output/empty_test",
)
```

**预期结果**:
- [ ] `results['total_images']` == 0
- [ ] 不抛出异常

#### TC-1.4 错误图片处理
**测试目标**: 验证单张图片错误不影响整体处理

**测试步骤**:
1. 在测试目录中放入一个损坏的图片文件
2. 运行批量推理

**预期结果**:
- [ ] 其他正常图片被正确处理
- [ ] 损坏图片的结果中包含 `'error'` 键

---

### 二、训练中断机制测试

#### TC-2.1 训练启动和进程创建
**测试目标**: 验证 `train_yolov8_stream_with_process()` 返回有效进程对象

**测试步骤**:
```python
from core.train_core import train_yolov8_stream_with_process
import time

proc, log_iter = train_yolov8_stream_with_process(
    data_yaml="data/cell.yaml",
    model_name="yolov8n.pt",
    epochs=2,
    imgsz=320,
    batch=4,
)

print(f"进程PID: {proc.pid}")
print(f"进程状态: {proc.poll()}")  # None表示运行中

# 读取几行日志
for i, line in enumerate(log_iter):
    print(line.strip())
    if i > 10:
        break
```

**预期结果**:
- [ ] `proc` 是有效的 `subprocess.Popen` 对象
- [ ] `proc.pid` 是有效的进程ID
- [ ] `proc.poll()` 返回 `None`（进程运行中）

#### TC-2.2 训练中断测试
**测试目标**: 验证可以成功终止训练进程

**测试步骤**:
```python
import os
import signal
import time

proc, log_iter = train_yolov8_stream_with_process(
    data_yaml="data/cell.yaml",
    epochs=100,  # 设置较长的训练
)

print(f"启动训练, PID: {proc.pid}")
time.sleep(10)  # 等待训练开始

# 终止进程
if os.name == 'nt':
    proc.terminate()
else:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

proc.wait(timeout=10)
print(f"进程退出码: {proc.returncode}")
```

**预期结果**:
- [ ] 进程在 10 秒内终止
- [ ] `proc.poll()` 返回非 `None` 值

#### TC-2.3 Gradio停止按钮测试 (手动)
**测试步骤**:
1. 启动 Gradio 界面: `python app.py`
2. 开始训练（epochs=100）
3. 等待看到训练日志输出
4. 点击"停止训练"按钮

**预期结果**:
- [ ] 显示 "✅ 训练已停止" 消息
- [ ] 训练进程确实终止（检查系统进程列表）

---

### 三、训练进度可视化测试

#### TC-3.1 进度解析测试
**测试目标**: 验证 `parse_training_progress()` 正确解析日志

**测试步骤**:
```python
from app import parse_training_progress

test_lines = [
    "Epoch 1/100",
    "Epoch 15/100",
    "Epoch 100/100",
    "Some other log line",
    "      Epoch 50/200      ",
]

for line in test_lines:
    result = parse_training_progress(line)
    print(f"'{line}' -> {result}")
```

**预期结果**:
- [ ] "Epoch 1/100" -> (1, 100)
- [ ] "Epoch 15/100" -> (15, 100)
- [ ] "Epoch 100/100" -> (100, 100)
- [ ] "Some other log line" -> None
- [ ] "Epoch 50/200" -> (50, 200)

#### TC-3.2 进度条显示测试 (手动)
**测试步骤**:
1. 启动 Gradio 界面
2. 开始训练
3. 观察日志输出区域

**预期结果**:
- [ ] 显示进度条 `[████░░░░░░░░░░░░░░░░]`
- [ ] 显示百分比 `xx.x%`
- [ ] 显示已用时间
- [ ] 显示预计剩余时间（ETA）
- [ ] 进度条随训练进度更新

---

### 四、标注工具增强测试

#### TC-4.1 撤销/重做功能测试 (手动)
**测试步骤**:
1. 运行推理获取检测结果
2. 点击"Open Annotator"打开标注工具
3. 添加一个新的标注框
4. 按 Ctrl+Z 撤销
5. 按 Ctrl+Y 重做

**预期结果**:
- [ ] 添加标注框后，框出现在画布上
- [ ] Ctrl+Z 后，新添加的框消失
- [ ] Ctrl+Y 后，框重新出现

#### TC-4.2 多次撤销测试
**测试步骤**:
1. 添加 3 个标注框
2. 连续按 Ctrl+Z 3 次
3. 连续按 Ctrl+Y 3 次

**预期结果**:
- [ ] 每次 Ctrl+Z 撤销一个操作
- [ ] 每次 Ctrl+Y 恢复一个操作
- [ ] 最终状态与初始添加后相同

#### TC-4.3 撤销栈边界测试
**测试步骤**:
1. 打开标注工具（不做任何操作）
2. 按 Ctrl+Z

**预期结果**:
- [ ] 不抛出异常
- [ ] 画面无变化

---

### 五、导出格式扩展测试

#### TC-5.1 COCO JSON导出测试
**测试步骤**:
```python
from core.export_utils import export_coco_json

image_paths = ["tests/test_images/img1.png", "tests/test_images/img2.png"]
all_detections = {
    "img1.png": [
        {"class_id": 0, "bbox": [100, 100, 200, 200], "confidence": 0.95},
        {"class_id": 1, "bbox": [300, 300, 400, 400], "confidence": 0.88},
    ],
    "img2.png": [
        {"class_id": 0, "bbox": [50, 50, 150, 150], "confidence": 0.92},
    ],
}

output_path = export_coco_json(
    image_paths=image_paths,
    all_detections=all_detections,
    output_path="tests/output/coco_export.json",
    class_names={0: "RBC", 1: "WBC"},
)

# 验证输出
import json
with open(output_path) as f:
    coco = json.load(f)

print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
print(f"Categories: {len(coco['categories'])}")
```

**预期结果**:
- [ ] 输出文件存在
- [ ] `coco['images']` 长度为 2
- [ ] `coco['annotations']` 长度为 3
- [ ] `coco['categories']` 包含所有类别
- [ ] bbox 格式为 [x, y, width, height]（COCO格式）

#### TC-5.2 Pascal VOC XML导出测试
**测试步骤**:
```python
from core.export_utils import export_pascal_voc_xml

detections = [
    {"class_id": 0, "bbox": [100, 100, 200, 200], "confidence": 0.95},
    {"class_id": 1, "bbox": [300, 300, 400, 400], "confidence": 0.88},
]

output_path = export_pascal_voc_xml(
    image_path="tests/test_images/img1.png",
    detections=detections,
    output_path="tests/output/voc_export.xml",
    class_names={0: "RBC", 1: "WBC"},
)

# 验证XML结构
import xml.etree.ElementTree as ET
tree = ET.parse(output_path)
root = tree.getroot()

print(f"Root tag: {root.tag}")
objects = root.findall('object')
print(f"Objects: {len(objects)}")
for obj in objects:
    print(f"  - {obj.find('name').text}")
```

**预期结果**:
- [ ] 输出文件存在
- [ ] XML根元素为 `annotation`
- [ ] 包含 2 个 `object` 元素
- [ ] 每个 object 有 `name`, `bndbox` 等子元素

---

### 六、Desktop UI测试 (手动)

#### TC-6.1 批量推理界面测试
**测试步骤**:
1. 运行 `python main.py`
2. 切换到 Inference 页面
3. 在 "Batch Inference" 区域填入输入/输出目录
4. 点击 "Run Batch Inference"

**预期结果**:
- [ ] 进度标签实时更新
- [ ] 完成后显示统计信息
- [ ] 按钮在处理期间禁用

#### TC-6.2 Gradio批量推理界面测试
**测试步骤**:
1. 运行 `python app.py`
2. 切换到"推理检测"标签
3. 在"批量推理"区域填入目录
4. 点击"运行批量推理"

**预期结果**:
- [ ] 进度区域显示处理状态
- [ ] 完成后显示详细统计

---

## 自动化测试脚本

创建 `tests/run_tests.py`:

```python
#!/usr/bin/env python
"""自动化测试脚本"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_batch_inference():
    """测试批量推理"""
    from core import infer_batch

    # 准备测试数据
    test_dir = ROOT / "tests" / "test_images"
    output_dir = ROOT / "tests" / "output" / "batch_auto"
    weights = ROOT / "runs" / "detect" / "cells" / "weights" / "best.pt"

    if not test_dir.exists():
        print(f"SKIP: 测试图片目录不存在: {test_dir}")
        return True

    if not weights.exists():
        print(f"SKIP: 模型权重不存在: {weights}")
        return True

    results = infer_batch(
        weights=str(weights),
        source_dir=str(test_dir),
        output_dir=str(output_dir),
    )

    assert results["total_images"] >= 0, "total_images should be >= 0"
    assert "csv_path" in results, "csv_path should be in results"

    print(f"PASS: 批量推理测试 - 处理 {results['total_images']} 张图片")
    return True


def test_progress_parsing():
    """测试进度解析"""
    # 导入时可能需要处理Gradio未安装的情况
    try:
        from app import parse_training_progress
    except ImportError:
        print("SKIP: 无法导入app模块")
        return True

    test_cases = [
        ("Epoch 1/100", (1, 100)),
        ("Epoch 50/200", (50, 200)),
        ("Some other line", None),
        ("", None),
    ]

    for line, expected in test_cases:
        result = parse_training_progress(line)
        assert result == expected, f"Failed for '{line}': got {result}, expected {expected}"

    print("PASS: 进度解析测试")
    return True


def test_export_coco():
    """测试COCO导出"""
    try:
        from core.export_utils import export_coco_json
    except ImportError:
        print("SKIP: export_utils模块不存在")
        return True

    import json
    import tempfile

    # 创建临时测试图片
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        Image.new("RGB", (100, 100), color="red").save(img_path)

        output_path = Path(tmpdir) / "coco.json"

        export_coco_json(
            image_paths=[str(img_path)],
            all_detections={
                "test.png": [{"class_id": 0, "bbox": [10, 10, 50, 50], "confidence": 0.9}]
            },
            output_path=str(output_path),
        )

        assert output_path.exists(), "COCO JSON file should exist"

        with open(output_path) as f:
            coco = json.load(f)

        assert len(coco["images"]) == 1, "Should have 1 image"
        assert len(coco["annotations"]) == 1, "Should have 1 annotation"

    print("PASS: COCO导出测试")
    return True


def test_export_voc():
    """测试VOC导出"""
    try:
        from core.export_utils import export_pascal_voc_xml
    except ImportError:
        print("SKIP: export_utils模块不存在")
        return True

    import tempfile
    import xml.etree.ElementTree as ET

    from PIL import Image
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        Image.new("RGB", (100, 100), color="red").save(img_path)

        output_path = Path(tmpdir) / "test.xml"

        export_pascal_voc_xml(
            image_path=str(img_path),
            detections=[{"class_id": 0, "bbox": [10, 10, 50, 50], "confidence": 0.9}],
            output_path=str(output_path),
        )

        assert output_path.exists(), "VOC XML file should exist"

        tree = ET.parse(output_path)
        root = tree.getroot()
        assert root.tag == "annotation", "Root should be 'annotation'"
        assert len(root.findall("object")) == 1, "Should have 1 object"

    print("PASS: VOC导出测试")
    return True


def main():
    """运行所有测试"""
    tests = [
        test_progress_parsing,
        test_export_coco,
        test_export_voc,
        test_batch_inference,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__} - {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"测试完成: {passed} 通过, {failed} 失败")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

---

## 测试执行命令

```bash
# 运行自动化测试
python tests/run_tests.py

# 单独测试批量推理
python -c "from core import infer_batch; print(infer_batch.__doc__)"

# 启动Gradio进行手动测试
python app.py

# 启动Desktop UI进行手动测试
python main.py
```

---

## 测试检查清单

### 功能完整性
- [ ] 批量推理函数 `infer_batch()` 存在并可调用
- [ ] 训练进程控制函数 `train_yolov8_stream_with_process()` 存在
- [ ] 进度解析函数 `parse_training_progress()` 存在
- [ ] 导出函数 `export_coco_json()` 和 `export_pascal_voc_xml()` 存在

### API导出
- [ ] `core/__init__.py` 导出 `infer_batch`
- [ ] `core/__init__.py` 导出 `train_yolov8_stream_with_process`
- [ ] `core/__init__.py` 导出 `export_coco_json` (如果有)
- [ ] `core/__init__.py` 导出 `export_pascal_voc_xml` (如果有)

### UI集成
- [ ] Gradio界面有批量推理区域
- [ ] Gradio训练页面显示进度条
- [ ] Desktop UI有批量推理区域
- [ ] 标注工具支持Ctrl+Z/Y
