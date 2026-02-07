"""
Microscopy AI - Gradio Web Interface
基于 Gradio 的显微镜细胞检测 Web 界面
"""

from __future__ import annotations

import os
import sys
import json
import time
import socket
import threading
from pathlib import Path
from collections import Counter
from typing import Iterator

import gradio as gr
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (
    scan_dataset,
    scan_dataset_from_yaml,
    build_val_split,
    infer_and_count,
    train_yolov8_stream,
    train_yolov8_stream_with_process,
    infer_batch,
    get_run_outputs,
    export_xanylabeling_json,
    save_dataset_report,
    load_class_mapping_rows,
    save_class_mapping_rows,
    rows_to_maps,
)
from core.constants import DEVICE_OPTIONS, MODEL_OPTIONS
from core.utils import resolve_device

# ============================================================================
# Constants & Config
# ============================================================================

DEFAULT_DATASET_ROOT = str(ROOT / "data")
DEFAULT_YAML_PATH = str(ROOT / "data" / "cell.yaml")
DEFAULT_OUTPUT_DIR = str(ROOT / "outputs" / "infer")
RUNS_DIR = str(ROOT / "runs" / "detect")
REPORTS_DIR = str(ROOT / "outputs" / "reports")

def _read_version() -> str:
    version_path = ROOT / "VERSION"
    if not version_path.exists():
        return "0.0.0"
    return version_path.read_text(encoding="utf-8").strip() or "0.0.0"


# Software version (single source of truth)
VERSION = _read_version()

# Shared constants live in core.constants to keep UI parity.

# Global state for training logs
training_logs: list[str] = []
is_training: bool = False
training_process = None

# Image cache for faster reloading
_image_cache: dict[str, Image.Image] = {}
_MAX_CACHE_SIZE = 5


def _get_cached_image(image_id: str, image_array: np.ndarray) -> Image.Image:
    """Get cached image or create new one."""
    cache_key = f"{image_id}_{id(image_array)}"
    if cache_key in _image_cache:
        return _image_cache[cache_key]

    # Clear cache if too large
    if len(_image_cache) >= _MAX_CACHE_SIZE:
        _image_cache.clear()

    pil_image = (
        Image.fromarray(image_array)
        if image_array.shape[2] == 3
        else Image.fromarray(image_array).convert("RGB")
    )
    _image_cache[cache_key] = pil_image
    return pil_image


# ============================================================================
# Helper Functions
# ============================================================================


def find_latest_model(runs_dir: str | Path) -> str | None:
    """Find the latest trained model in runs directory."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None

    best_candidates = []
    for subdir in runs_path.rglob("weights"):
        best_pt = subdir / "best.pt"
        if best_pt.exists():
            best_candidates.append((best_pt.stat().st_mtime, str(best_pt)))

    if not best_candidates:
        return None

    best_candidates.sort(reverse=True)
    return best_candidates[0][1]


def find_available_port(host: str, start: int = 7860, end: int = 7959) -> int | None:
    """Find an available TCP port on host within [start, end], then fallback to ephemeral."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, 0))
        except OSError:
            return None
        return int(sock.getsockname()[1])


def format_dataset_stats(stats: dict) -> str:
    """Format dataset statistics for display."""
    _, cn_map = get_active_class_maps()
    lines = [
        f"📁 Dataset root: {stats.get('dataset_root', '')}",
        f"📂 Layout: {stats.get('layout', 'unknown')}",
        f"🖼️ Total images: {stats.get('total_images', 0)}",
        f"🏷️ Total labels: {stats.get('total_labels', 0)}",
        f"❌ Missing labels: {stats.get('missing_labels', 0)}",
        f"❌ Missing images: {stats.get('missing_images', 0)}",
        "",
        "📊 Split stats:",
    ]
    for split, s in stats.get("split_stats", {}).items():
        lines.append(
            f"  • {split}: images={s.get('images', 0)}, labels={s.get('labels', 0)}, "
            f"missing_labels={s.get('missing_labels', 0)}"
        )

    lines.append("")
    lines.append("📈 Class counts:")
    class_counts = stats.get("class_counts", {})
    if class_counts:
        for k, v in sorted(class_counts.items(), key=lambda x: int(x[0])):
            cn_name = cn_map.get(int(k), f"Class {k}")
            lines.append(f"  • {cn_name} (ID {k}): {v}")
    else:
        lines.append("  • none")

    return "\n".join(lines)


def get_active_class_maps() -> tuple[dict[int, str], dict[int, str]]:
    rows = load_class_mapping_rows(ROOT)
    return rows_to_maps(rows)


def class_mapping_rows() -> list[list[object]]:
    rows = load_class_mapping_rows(ROOT)
    return [[row["id"], row.get("en", ""), row.get("cn", "")] for row in rows]


def save_class_mapping_table(table: list[list[object]]) -> tuple[str, list[list[object]]]:
    by_id: dict[int, dict[str, object]] = {}
    for row in table or []:
        if not row or len(row) < 1:
            continue
        try:
            cid = int(row[0])
        except Exception:
            continue
        en = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
        cn = str(row[2]).strip() if len(row) > 2 and row[2] is not None else ""
        by_id[cid] = {"id": cid, "en": en, "cn": cn}
    if not by_id:
        return "❌ Error: No valid rows to save.", class_mapping_rows()
    path = save_class_mapping_rows(ROOT, by_id.values())
    return f"✅ Saved: {path}", class_mapping_rows()


def delete_class_mapping_row(
    table: list[list[object]], delete_id: int | float | str
) -> tuple[str, list[list[object]]]:
    try:
        cid = int(delete_id)
    except Exception:
        return "❌ Error: Invalid ID.", table or []
    new_rows: list[list[object]] = []
    removed = False
    for row in table or []:
        if not row or len(row) < 1:
            continue
        try:
            row_id = int(row[0])
        except Exception:
            continue
        if row_id == cid:
            removed = True
            continue
        en = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
        cn = str(row[2]).strip() if len(row) > 2 and row[2] is not None else ""
        new_rows.append([row_id, en, cn])
    if not removed:
        return f"⚠️ ID {cid} not found.", table or []
    return f"✅ Removed ID {cid}.", new_rows


def get_class_distribution_chart(stats: dict) -> np.ndarray | None:
    """Generate a bar chart for class distribution using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        class_counts = stats.get("class_counts", {})
        if not class_counts:
            return None

        # Sort by class ID
        items = sorted(class_counts.items(), key=lambda x: int(x[0]))
        ids = [int(k) for k, v in items]
        counts = [v for k, v in items]
        _, cn_map = get_active_class_maps()
        labels = [f"{cn_map.get(i, str(i))}\n(ID:{i})" for i in ids]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(
            range(len(ids)), counts, color="#3B82F6", edgecolor="#1D4ED8", linewidth=1.5
        )

        # Customize
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Cell Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Dataset Class Distribution", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        # Convert to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return buf
    except Exception as e:
        print(f"Chart generation error: {e}")
        return None


# ============================================================================
# Dashboard Functions
# ============================================================================


def scan_dataset_handler(dataset_root: str) -> tuple[str, np.ndarray | None]:
    """Handle dataset scanning."""
    try:
        if not dataset_root or not Path(dataset_root).exists():
            return "❌ Error: Dataset root does not exist.", None

        stats = scan_dataset(dataset_root)
        _, cn_map = get_active_class_maps()
        report_path = save_dataset_report(stats, REPORTS_DIR, class_name_map=cn_map)
        text_output = format_dataset_stats(stats) + f"\n\nReport saved: {report_path}"
        chart = get_class_distribution_chart(stats)
        return text_output, chart
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def scan_from_yaml(yaml_path: str) -> tuple[str, np.ndarray | None]:
    """Scan dataset from YAML config."""
    try:
        if not yaml_path or not Path(yaml_path).exists():
            return "❌ Error: YAML file does not exist.", None

        stats = scan_dataset_from_yaml(yaml_path)
        _, cn_map = get_active_class_maps()
        report_path = save_dataset_report(stats, REPORTS_DIR, class_name_map=cn_map)
        text_output = format_dataset_stats(stats) + f"\n\nReport saved: {report_path}"
        chart = get_class_distribution_chart(stats)
        return text_output, chart
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def regenerate_val_split(dataset_root: str, ratio: float = 0.1) -> str:
    """Regenerate validation split from training data."""
    try:
        if not dataset_root or not Path(dataset_root).exists():
            return "❌ Error: Dataset root does not exist."

        stats = build_val_split(dataset_root, ratio=ratio, seed=42)
        if stats["val_images"] == 0:
            return "❌ Error: Cannot regenerate val: no train images found."

        return (
            f"✅ Successfully regenerated validation split:\n"
            f"   • Train images: {stats['train_images']}\n"
            f"   • Val images: {stats['val_images']} ({ratio * 100:.0f}%)"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Training Functions
# ============================================================================


def check_dataset_handler(yaml_path: str) -> str:
    """Check if dataset is valid for training."""
    try:
        stats = scan_dataset_from_yaml(yaml_path)
        split_stats = stats.get("split_stats", {})
        val = split_stats.get("val", {})
        train = split_stats.get("train", {})

        messages = [
            f"📊 Dataset check results:",
            f"   • Train images: {train.get('images', 0)}",
            f"   • Val images: {val.get('images', 0)}",
        ]

        if train.get("images", 0) == 0:
            messages.append("\n❌ ERROR: No training images found!")
            return "\n".join(messages)

        if val.get("images", 0) == 0:
            messages.append(
                "\n⚠️ WARNING: Validation images missing. Please regenerate val split."
            )
            return "\n".join(messages)

        messages.append("\n✅ Dataset check passed!")
        return "\n".join(messages)

    except Exception as e:
        return f"❌ Error: {str(e)}"


def parse_training_progress(line: str) -> tuple[int, int] | None:
    """从训练日志中解析进度 (current_epoch, total_epochs)"""
    import re

    match = re.search(r"Epoch\s+(\d+)/(\d+)", line)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def train_model_stream(
    data_yaml: str,
    model_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    limit_train: int,
    limit_val: int,
) -> Iterator[str]:
    """Stream training logs."""
    global is_training, training_logs, training_process

    if not data_yaml or not Path(data_yaml).exists():
        yield "❌ Error: Dataset YAML does not exist."
        return

    is_training = True
    training_logs = []
    start_time = time.time()

    yield "🚀 Starting training...\n" + "=" * 50 + "\n"

    try:
        proc, log_iterator = train_yolov8_stream_with_process(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            limit_train_images=limit_train if limit_train > 0 else 0,
            limit_val_images=limit_val if limit_val > 0 else 0,
            remap_classes=False,
        )
        training_process = proc
        current_epoch = 0
        total_epochs = epochs

        for line in log_iterator:
            if not is_training:
                break
            # Clean ANSI codes
            import re

            clean_line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
            training_logs.append(clean_line)
            elapsed = time.time() - start_time
            progress = parse_training_progress(clean_line)
            if progress:
                current_epoch, total_epochs = progress

            if current_epoch > 0:
                eta_seconds = (elapsed / current_epoch) * (total_epochs - current_epoch)
                eta_str = f"{int(eta_seconds // 60)}分{int(eta_seconds % 60)}秒"
            else:
                eta_str = "计算中..."

            progress_pct = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0
            progress_bar = f"[{'█' * int(progress_pct // 5)}{'░' * (20 - int(progress_pct // 5))}]"

            header = (
                f"📊 训练进度: {progress_bar} {progress_pct:.1f}%\n"
                f"⏱️ 已用时: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}\n"
                f"⏳ 预计剩余: {eta_str}\n"
                f"📈 当前轮次: {current_epoch}/{total_epochs}\n"
                f"{'=' * 50}\n"
            )
            yield header + "\n".join(training_logs[-30:])

        outputs = get_run_outputs("runs/detect", "cells")
        best_path = outputs.get("best", "")

        yield (
            f"\n✅ Training completed!\n"
            f"{'=' * 50}\n"
            f"📁 Model saved to: {best_path}\n"
            f"⏱️ Total time: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        )

    except Exception as e:
        yield f"\n❌ Training error: {str(e)}"
    finally:
        is_training = False
        training_process = None


def stop_training() -> str:
    """真正终止训练进程"""
    global is_training, training_process
    is_training = False
    if training_process is not None:
        try:
            import signal
            import os

            if os.name == "nt":
                training_process.terminate()
            else:
                os.killpg(os.getpgid(training_process.pid), signal.SIGTERM)
            training_process.wait(timeout=5)
            return "✅ 训练已停止"
        except Exception as e:
            try:
                training_process.kill()
                return f"⚠️ 训练已强制终止: {e}"
            except Exception:
                return f"❌ 无法终止训练进程: {e}"

    return "⚠️ 没有正在运行的训练任务"


# ============================================================================
# Inference Functions
# ============================================================================


def preprocess_image(
    image: np.ndarray, max_size: int = 1920, fast_mode: bool = False
) -> Image.Image:
    """Preprocess image: convert and optionally resize for faster processing.

    Args:
        image: Input numpy array
        max_size: Maximum dimension (0 = no limit)
        fast_mode: If True, uses faster but lower quality resize
    """
    if image is None:
        return None

    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        if image.shape[2] == 3:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image).convert("RGB")
    else:
        pil_image = image

    # Resize if image is too large (speeds up both saving and inference)
    if max_size > 0:
        w, h = pil_image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            # Use faster resampling in fast mode
            resample = (
                Image.Resampling.BILINEAR if fast_mode else Image.Resampling.LANCZOS
            )
            pil_image = pil_image.resize(new_size, resample)

    return pil_image


def get_image_info(image: np.ndarray) -> str:
    """Get image size info for display."""
    if image is None:
        return "未选择图片"

    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        mp = (w * h) / 1000000  # Megapixels
        size_info = f"{w}×{h} ({mp:.1f}MP)"

        # Estimate processing speed
        if max(w, h) > 4000:
            return f"📷 {size_info} - 🔴 大尺寸图片，建议启用压缩以加快处理"
        elif max(w, h) > 2500:
            return f"📷 {size_info} - 🟡 中等尺寸，建议适度压缩"
        else:
            return f"📷 {size_info} - 🟢 尺寸适中，可直接处理"

    return "未知尺寸"


def run_inference(
    image: np.ndarray,
    weights_path: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    use_chinese_labels: bool,
    preprocess_max_size: int = 1920,
) -> tuple[Image.Image, str]:
    """Run inference on an image with preprocessing optimization."""
    try:
        if image is None:
            return None, "❌ Error: No image provided."

        if not weights_path or not Path(weights_path).exists():
            return None, f"❌ Error: Weights not found: {weights_path}"

        # Determine quality settings based on max_size
        fast_mode = preprocess_max_size <= 1280 and preprocess_max_size > 0
        jpeg_quality = 85 if fast_mode else 95

        # Preprocess image (resize if too large)
        pil_image = preprocess_image(
            image, max_size=preprocess_max_size, fast_mode=fast_mode
        )
        if pil_image is None:
            return None, "❌ Error: Failed to process image."

        # Use JPEG for temporary file (faster than PNG)
        temp_path = ROOT / "temp_infer_image.jpg"
        pil_image.save(temp_path, quality=jpeg_quality, optimize=True)

        # Determine label mapping
        _, cn_map = get_active_class_maps()
        label_mapping = cn_map if use_chinese_labels else None

        # Run inference
        vis_img, counts, total, dets = infer_and_count(
            weights=weights_path,
            source_image=str(temp_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            return_dets=True,
            label_mapping=label_mapping,
        )

        # Clean up temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
        except:
            pass

        # Optimize output image for web display
        # Limit output size to reduce transfer time
        out_w, out_h = vis_img.size
        max_display_size = 1600
        if max(out_w, out_h) > max_display_size:
            ratio = max_display_size / max(out_w, out_h)
            new_size = (int(out_w * ratio), int(out_h * ratio))
            vis_img = vis_img.resize(new_size, Image.Resampling.LANCZOS)

        # Format results
        result_lines = [f"🎯 Total cells detected: {total}", "", "📊 Per-class counts:"]
        _, cn_map = get_active_class_maps()
        for cls_id, count in sorted(counts.items()):
            name = cn_map.get(cls_id, f"Class {cls_id}")
            result_lines.append(f"   • {name} (ID {cls_id}): {count}")

        return vis_img, "\n".join(result_lines)

    except Exception as e:
        import traceback

        return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"


def auto_find_weights() -> str:
    """Auto-find the latest trained weights."""
    latest = find_latest_model(RUNS_DIR)
    if latest:
        return latest
    # Fallback to pretrained
    for model in ["yolo11n.pt", "yolov8n.pt"]:
        path = ROOT / model
        if path.exists():
            return str(path)
    return ""


def export_to_xanylabeling(
    image: np.ndarray,
    custom_filename: str,
    weights_path: str,
    output_dir: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    preprocess_max_size: int = 1920,
) -> str:
    """Export inference results to X-AnyLabeling format with optimization.

    Args:
        image: Input image array
        custom_filename: Custom filename for export (without extension)
        ... (other params)
    """
    try:
        if image is None:
            return "❌ Error: No image provided."

        if not weights_path or not Path(weights_path).exists():
            return f"❌ Error: Weights not found: {weights_path}"

        # Determine quality settings
        fast_mode = preprocess_max_size <= 1280 and preprocess_max_size > 0
        jpeg_quality = 85 if fast_mode else 95

        # Preprocess image
        pil_image = preprocess_image(
            image, max_size=preprocess_max_size, fast_mode=fast_mode
        )
        if pil_image is None:
            return "❌ Error: Failed to process image."

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use custom filename or timestamp
        if custom_filename and custom_filename.strip():
            # Sanitize filename
            base_name = "".join(
                c for c in custom_filename.strip() if c.isalnum() or c in "-_"
            ).strip()
            if not base_name:
                base_name = "export"
        else:
            # Use timestamp
            from datetime import datetime

            base_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        image_path = out_dir / f"{base_name}.jpg"
        pil_image.save(image_path, quality=jpeg_quality, optimize=True)

        # Run inference with detections
        _, cn_map = get_active_class_maps()
        vis_img, counts, total, dets = infer_and_count(
            weights=weights_path,
            source_image=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            return_dets=True,
            label_mapping=cn_map,
        )

        # Export JSON with same base name
        output_json = out_dir / f"{base_name}.json"
        export_xanylabeling_json(
            image_path=str(image_path),
            image_size=pil_image.size,
            detections=dets,
            output_json=str(output_json),
            label_mapping=cn_map,
        )

        return f"✅ Exported to:\n   • Image: {image_path.name}\n   • JSON: {output_json.name}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def run_batch_inference(
    input_dir: str,
    output_dir: str,
    weights_path: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> Iterator[tuple[str, str]]:
    """批量推理生成器"""
    if not input_dir or not Path(input_dir).exists():
        yield "❌ 输入目录不存在", ""
        return

    _, cn_map = get_active_class_maps()

    yield "🚀 开始批量推理...", ""

    try:
        results = infer_batch(
            weights=weights_path,
            source_dir=input_dir,
            output_dir=output_dir,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            label_mapping=cn_map,
        )

        summary = (
            "✅ 批量推理完成!\n\n"
            "📊 统计:\n"
            f"- 处理图片数: {results['total_images']}\n"
            f"- 检测细胞总数: {results['total_cells']}\n"
            f"- CSV报告: {results['csv_path']}\n\n"
            "📈 类别统计:"
        )
        for cls_id, count in sorted(results["summary"].items()):
            name = cn_map.get(cls_id, f"Class {cls_id}")
            summary += f"\n- {name}: {count}"

        yield "✅ 完成", summary

    except Exception as e:
        yield f"❌ 错误: {str(e)}", ""


# ============================================================================
# Gradio Interface
# ============================================================================


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Microscopy AI - Cell Detection",
    ) as demo:
        # Apply custom CSS
        gr.Markdown(
            """
            <style>
            :root { color-scheme: dark; }
            body, .gradio-container {
              background: #0a0e1a !important;
              color: #e5e7eb !important;
            }
            .gradio-container {
              font-family: "Inter", "SF Pro Text", "Segoe UI", system-ui, sans-serif;
            }
            .tab-item { font-size: 16px !important; font-weight: 500 !important; }
            .output-text { font-family: "JetBrains Mono", Consolas, monospace; white-space: pre-wrap; }
            .gradio-container .tabs { border-bottom: 1px solid #30363d; }
            .gradio-container .tab-nav button {
              background: transparent !important;
              color: #9ca3af !important;
              border: 0 !important;
              padding: 10px 14px !important;
            }
            .gradio-container .tab-nav button.selected {
              color: #60a5fa !important;
              border-bottom: 2px solid #3b82f6 !important;
            }
            .card {
              background: #0d1117 !important;
              border: 1px solid #30363d !important;
              border-radius: 10px !important;
              padding: 16px !important;
              box-shadow: none !important;
            }
            input, textarea, select {
              background: #161b22 !important;
              border: 1px solid #374151 !important;
              color: #e5e7eb !important;
              border-radius: 8px !important;
            }
            input::placeholder, textarea::placeholder { color: #6b7280 !important; }
            button {
              border-radius: 8px !important;
              border: 1px solid #374151 !important;
              background: #161b22 !important;
              color: #e5e7eb !important;
            }
            .primary button, button.primary {
              background: linear-gradient(90deg, #f97316, #ea580c) !important;
              border: 0 !important;
              color: #fff !important;
            }
            .accent button, button.accent {
              background: #21262d !important;
              border: 1px solid #374151 !important;
              color: #cbd5f5 !important;
            }
            </style>
            """
        )

        gr.Markdown(
            """
            # 🔬 Microscopy AI - 细胞检测系统
            基于 YOLOv8 的显微镜血细胞检测、分类与计数
            """
        )

        # ========================================================================
        # Tab 1: Dashboard
        # ========================================================================
        with gr.Tab("📊 数据仪表板"):
            gr.Markdown("### 数据集概览与统计")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        dataset_root_input = gr.Textbox(
                            label="数据集根目录",
                            value=DEFAULT_DATASET_ROOT,
                            placeholder="输入数据集根目录路径",
                        )
                        yaml_path_input = gr.Textbox(
                            label="YAML 配置文件路径",
                            value=DEFAULT_YAML_PATH,
                            placeholder="输入 YAML 文件路径",
                        )

                        with gr.Row():
                            scan_btn = gr.Button("🔍 扫描数据集", variant="primary", elem_classes=["primary"])
                            scan_yaml_btn = gr.Button(
                                "🔍 从 YAML 扫描", variant="secondary", elem_classes=["accent"]
                            )

                        with gr.Row():
                            regen_val_btn = gr.Button(
                                "🔄 重新生成验证集", variant="secondary", elem_classes=["accent"]
                            )
                            val_ratio = gr.Slider(
                                label="验证集比例",
                                minimum=0.05,
                                maximum=0.5,
                                value=0.1,
                                step=0.05,
                            )

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card"]):
                        dataset_stats_output = gr.Textbox(
                            label="数据集统计",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            elem_classes=["output-text"],
                        )

            with gr.Row():
                with gr.Group(elem_classes=["card"]):
                    class_dist_plot = gr.Image(
                        label="类别分布图",
                        type="numpy",
                        interactive=False,
                    )

            regen_val_output = gr.Textbox(label="验证集生成结果", interactive=False)

            # Event handlers
            scan_btn.click(
                fn=scan_dataset_handler,
                inputs=[dataset_root_input],
                outputs=[dataset_stats_output, class_dist_plot],
            )
            scan_yaml_btn.click(
                fn=scan_from_yaml,
                inputs=[yaml_path_input],
                outputs=[dataset_stats_output, class_dist_plot],
            )
            regen_val_btn.click(
                fn=regenerate_val_split,
                inputs=[dataset_root_input, val_ratio],
                outputs=[regen_val_output],
            )

        # ========================================================================
        # Tab 2: Training
        # ========================================================================
        with gr.Tab("🚀 模型训练"):
            gr.Markdown("### 训练 YOLOv8 模型")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 数据集与模型配置")
                        train_yaml_input = gr.Textbox(
                            label="数据集 YAML 文件",
                            value=DEFAULT_YAML_PATH,
                        )
                        model_select = gr.Dropdown(
                            label="预训练模型",
                            choices=MODEL_OPTIONS,
                            value="yolov8n.pt",
                        )
                        device_select = gr.Dropdown(
                            label="计算设备",
                            choices=DEVICE_OPTIONS,
                            value="auto",
                        )

                        check_dataset_btn = gr.Button("✅ 检查数据集", variant="secondary", elem_classes=["accent"])
                        check_output = gr.Textbox(
                            label="检查结果", lines=5, interactive=False
                        )

                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 训练参数")
                        epochs_input = gr.Slider(
                            label="训练轮数 (Epochs)",
                            minimum=1,
                            maximum=1000,
                            value=100,
                            step=10,
                        )
                        batch_input = gr.Slider(
                            label="批次大小 (Batch Size)",
                            minimum=1,
                            maximum=128,
                            value=16,
                            step=1,
                        )
                        imgsz_input = gr.Slider(
                            label="图像尺寸 (Image Size)",
                            minimum=320,
                            maximum=1280,
                            value=640,
                            step=32,
                        )

                        with gr.Row():
                            limit_train_input = gr.Number(
                                label="限制训练图片数 (0=全部)",
                                value=0,
                                minimum=0,
                            )
                            limit_val_input = gr.Number(
                                label="限制验证图片数 (0=全部)",
                                value=0,
                                minimum=0,
                            )

            with gr.Row():
                start_train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg", elem_classes=["primary"])
                stop_train_btn = gr.Button("⏹️ 停止训练", variant="stop", size="lg", elem_classes=["accent"])

            with gr.Group(elem_classes=["card"]):
                train_logs_output = gr.Textbox(
                    label="训练日志",
                    lines=30,
                    max_lines=50,
                    interactive=False,
                    elem_classes=["output-text"],
                    autoscroll=True,
                )

            # Event handlers
            check_dataset_btn.click(
                fn=check_dataset_handler,
                inputs=[train_yaml_input],
                outputs=[check_output],
            )
            start_train_btn.click(
                fn=train_model_stream,
                inputs=[
                    train_yaml_input,
                    model_select,
                    epochs_input,
                    batch_input,
                    imgsz_input,
                    device_select,
                    limit_train_input,
                    limit_val_input,
                ],
                outputs=[train_logs_output],
            )
            stop_train_btn.click(
                fn=stop_training,
                outputs=[train_logs_output],
            )

        # ========================================================================
        # Tab 3: Inference
        # ========================================================================
        with gr.Tab("🔍 推理检测"):
            gr.Markdown("### 细胞检测与计数")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 输入与参数")
                        input_image = gr.Image(
                            label="上传显微镜图像 (支持拖拽)",
                            type="numpy",
                            image_mode="RGB",
                            height=400,  # 限制预览高度
                            streaming=False,  # Disable streaming for faster upload
                        )

                        # Custom filename for export (defaults to timestamp)
                        export_filename = gr.Textbox(
                            label="导出文件名 (不含扩展名)",
                            value="",
                            placeholder="留空则使用当前时间戳",
                            info="导出时会自动添加 .jpg 和 .json 扩展名",
                        )

                        image_info = gr.Textbox(
                            label="图片信息",
                            value="未选择图片",
                            interactive=False,
                        )
                        with gr.Row():
                            preprocess_size = gr.Slider(
                                label="最大边长限制 (0=不限制, 推荐1920)",
                                info="大尺寸图片会被自动缩放，加快处理速度",
                                minimum=0,
                                maximum=4096,
                                value=1920,
                                step=64,
                            )

                        speed_mode = gr.Radio(
                            label="处理模式",
                            choices=[
                                ("⚡ 极速模式 (最大1280, JPEG 85%)", 1280),
                                ("🚀 平衡模式 (最大1920, JPEG 95%)", 1920),
                                ("🎯 质量模式 (不压缩)", 0),
                            ],
                            value=1920,
                        )

                        weights_input = gr.Textbox(
                            label="模型权重路径",
                            value=auto_find_weights(),
                            placeholder="选择 .pt 权重文件",
                        )
                        auto_find_btn = gr.Button(
                            "🔍 自动查找最新模型", size="sm", elem_classes=["accent"]
                        )

                        with gr.Row():
                            infer_imgsz = gr.Slider(
                                label="图像尺寸",
                                minimum=320,
                                maximum=1280,
                                value=640,
                                step=32,
                            )
                            infer_conf = gr.Slider(
                                label="置信度阈值",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.25,
                                step=0.01,
                            )

                        with gr.Row():
                            infer_iou = gr.Slider(
                                label="IoU 阈值",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.45,
                                step=0.01,
                            )
                            infer_device = gr.Dropdown(
                                label="计算设备",
                                choices=DEVICE_OPTIONS,
                                value="auto",
                            )

                        use_cn_labels = gr.Checkbox(
                            label="使用中文类别名称",
                            value=True,
                        )

                        gr.Markdown("💡 **提示**: 如果图片加载慢，请调小『最大边长限制』")

                        run_infer_btn = gr.Button(
                            "🔍 运行推理", variant="primary", size="lg", elem_classes=["primary"]
                        )

                    # Link speed mode to preprocess_size
                    def update_preprocess_size(mode_value):
                        return gr.update(value=mode_value)

                    speed_mode.change(
                        fn=update_preprocess_size,
                        inputs=[speed_mode],
                        outputs=[preprocess_size],
                    )

                    # Update image info when image changes
                    input_image.change(
                        fn=get_image_info,
                        inputs=[input_image],
                        outputs=[image_info],
                    )

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 检测结果")
                        output_image = gr.Image(
                            label="检测结果",
                            type="pil",
                            interactive=False,
                        )
                        inference_results = gr.Textbox(
                            label="检测统计",
                            lines=25,
                            interactive=False,
                            elem_classes=["output-text"],
                        )

            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 导出到 X-AnyLabeling")
                        export_dir = gr.Textbox(
                            label="输出目录",
                            value=str(ROOT / "outputs" / "xanylabeling"),
                        )
                        export_btn = gr.Button("📤 导出标注文件", variant="secondary", elem_classes=["accent"])
                        export_output = gr.Textbox(label="导出结果", interactive=False)

            with gr.Row():
                with gr.Group(elem_classes=["card"]):
                    gr.Markdown("#### 批量推理")
                    batch_input_dir = gr.Textbox(
                        label="输入目录",
                        placeholder="选择包含图片的目录",
                    )
                    batch_output_dir = gr.Textbox(
                        label="输出目录",
                        value=str(ROOT / "outputs" / "batch_infer"),
                    )
                    batch_progress = gr.Textbox(
                        label="进度",
                        value="等待开始...",
                        interactive=False,
                    )
                    run_batch_btn = gr.Button("🚀 运行批量推理", variant="primary", elem_classes=["primary"])
                    batch_results = gr.Textbox(
                        label="批量推理结果",
                        lines=10,
                        interactive=False,
                    )

            # Event handlers
            auto_find_btn.click(
                fn=lambda: auto_find_weights(),
                outputs=[weights_input],
            )
            run_infer_btn.click(
                fn=run_inference,
                inputs=[
                    input_image,
                    weights_input,
                    infer_imgsz,
                    infer_conf,
                    infer_iou,
                    infer_device,
                    use_cn_labels,
                    preprocess_size,
                ],
                outputs=[output_image, inference_results],
            )
            # Capture filename on upload (using a workaround with upload event)
            # Note: Gradio doesn't directly expose filename, so we use a default
            # The filename will be shown in the export output for verification

            export_btn.click(
                fn=export_to_xanylabeling,
                inputs=[
                    input_image,
                    export_filename,
                    weights_input,
                    export_dir,
                    infer_imgsz,
                    infer_conf,
                    infer_iou,
                    infer_device,
                    preprocess_size,
                ],
                outputs=[export_output],
            )

            run_batch_btn.click(
                fn=run_batch_inference,
                inputs=[
                    batch_input_dir,
                    batch_output_dir,
                    weights_input,
                    infer_imgsz,
                    infer_conf,
                    infer_iou,
                    infer_device,
                ],
                outputs=[batch_progress, batch_results],
            )

        # ========================================================================
        # Tab 4: Settings
        # ========================================================================
        with gr.Tab("⚙️ 设置"):
            gr.Markdown("### 系统设置")

            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 系统信息")
                        device_info = gr.Textbox(
                            label="设备信息",
                            value=f"PyTorch: {resolve_device('auto')}",
                            interactive=False,
                        )

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 类别名称映射")
                        class_mapping_table = gr.Dataframe(
                            headers=["ID", "English", "中文"],
                            datatype=["number", "str", "str"],
                            value=class_mapping_rows(),
                            row_count=(len(class_mapping_rows()), "dynamic"),
                            col_count=(3, "fixed"),
                            interactive=True,
                            label="类别映射（可编辑）",
                        )
                        delete_id = gr.Number(label="删除行 ID", value=None)
                        delete_btn = gr.Button("🗑️ 删除行", variant="secondary", elem_classes=["accent"])
                        save_mapping_btn = gr.Button("💾 保存映射", variant="primary", elem_classes=["primary"])
                        reset_mapping_btn = gr.Button("↩️ 恢复默认映射", variant="secondary", elem_classes=["accent"])
                        mapping_status = gr.Textbox(label="状态", interactive=False)

                    save_mapping_btn.click(
                        fn=save_class_mapping_table,
                        inputs=[class_mapping_table],
                        outputs=[mapping_status, class_mapping_table],
                    )
                    delete_btn.click(
                        fn=delete_class_mapping_row,
                        inputs=[class_mapping_table, delete_id],
                        outputs=[mapping_status, class_mapping_table],
                    )
                    reset_mapping_btn.click(
                        fn=lambda: ("✅ 已恢复默认映射（未保存）", class_mapping_rows()),
                        outputs=[mapping_status, class_mapping_table],
                    )

        gr.Markdown(
            f"""
            ---
            <center>
            <small>Microscopy AI System v{VERSION} | Powered by YOLOv8 & Gradio</small>
            </center>
            """
        )

    return demo


# ============================================================================
# Main Entry
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()

    # Enable queue for better concurrency handling
    demo.queue(max_size=20)

    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port_env = os.getenv("GRADIO_SERVER_PORT")
    launch_kwargs = {
        "server_name": server_name,
        "share": False,
        "show_error": True,
    }
    if server_port_env:
        launch_kwargs["server_port"] = int(server_port_env)
    else:
        available_port = find_available_port(server_name)
        if available_port is not None:
            launch_kwargs["server_port"] = available_port

    demo.launch(**launch_kwargs)
