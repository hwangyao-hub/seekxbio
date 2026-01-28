from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from core import (
    get_run_outputs,
    infer_and_count,
    scan_dataset,
    train_yolov8_stream,
)


def format_dataset_summary(stats: dict[str, Any]) -> str:
    lines = [
        f"Dataset root: {stats.get('dataset_root', '')}",
        f"Layout: {stats.get('layout', 'unknown')}",
        f"Total images: {stats.get('total_images', 0)}",
        f"Total labels: {stats.get('total_labels', 0)}",
        f"Missing labels: {stats.get('missing_labels', 0)}",
        f"Missing images: {stats.get('missing_images', 0)}",
        "",
        "Split stats:",
    ]
    split_stats = stats.get("split_stats", {})
    for split, s in split_stats.items():
        lines.append(
            f"- {split}: images={s.get('images', 0)}, labels={s.get('labels', 0)}, "
            f"missing_labels={s.get('missing_labels', 0)}, missing_images={s.get('missing_images', 0)}"
        )
    lines.append("")
    lines.append("Class counts:")
    class_counts = stats.get("class_counts", {})
    if class_counts:
        for k, v in class_counts.items():
            lines.append(f"- class {k}: {v}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def ui_scan_dataset(dataset_root: str) -> tuple[str, dict[str, Any]]:
    try:
        stats = scan_dataset(dataset_root)
        summary = format_dataset_summary(stats)
        return summary, stats
    except Exception as exc:
        return f"Error: {exc}", {}


def ui_train_model(
    data_yaml: str,
    model_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    limit_train: int,
    limit_val: int,
) -> tuple[str, dict[str, str]]:
    try:
        outputs = train_yolov8(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            limit_train_images=int(limit_train),
            limit_val_images=int(limit_val),
        )
        msg_lines = [
            "Training finished.",
            f"Save dir: {outputs.get('save_dir', '')}",
            f"Best: {outputs.get('best', '')}",
            f"Last: {outputs.get('last', '')}",
        ]
        return "\n".join(msg_lines), outputs
    except Exception as exc:
        return f"Error: {exc}", {}


def ui_train_model_stream(
    data_yaml: str,
    model_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    limit_train: int,
    limit_val: int,
) -> tuple[str, dict[str, str], str]:
    log = ""
    status = "Running..."
    for line in train_yolov8_stream(
        data_yaml=data_yaml,
        model_name=model_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        limit_train_images=int(limit_train),
        limit_val_images=int(limit_val),
    ):
        log += line
        yield log, {}, status

    outputs = get_run_outputs("runs/detect", "cells")
    status = "Training finished."
    yield log, outputs, status


def ui_infer_count(
    weights: str,
    image_path: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> tuple[Any, dict[str, int], str]:
    try:
        vis_img, counts, total = infer_and_count(
            weights=weights,
            source_image=image_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
        )
        total_text = f"Total cells: {total}"
        return vis_img, counts, total_text
    except Exception as exc:
        return None, {}, f"Error: {exc}"


def ui_find_latest_model() -> tuple[str, str]:
    try:
        base = Path("runs") / "detect"
        if not base.exists():
            return "", "No runs/detect directory found."
        candidates = list(base.rglob("best.pt"))
        if not candidates:
            return "", "No best.pt found under runs/detect."
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest), f"Latest model: {latest}"
    except Exception as exc:
        return "", f"Error: {exc}"


def build_app() -> gr.Blocks:
    css = """
    :root {
      --bg: #f4f7f5;
      --panel: #ffffff;
      --ink: #1f2a33;
      --muted: #6b7b86;
      --accent: #0b6b5d;
      --accent-weak: #d7ebe6;
      --border: #d9e2e6;
    }
    body, .gradio-container { background: var(--bg); color: var(--ink); }
    .gradio-container { font-family: "Segoe UI", "Noto Sans", "Helvetica Neue", Arial, sans-serif; }
    .block { border: 1px solid var(--border); border-radius: 10px; }
    .panel { background: var(--panel); }
    .muted { color: var(--muted); font-size: 12px; }
    .primary-btn button {
      background: var(--accent) !important;
      color: #fff !important;
      border: 1px solid var(--accent) !important;
      font-weight: 600;
      letter-spacing: 0.2px;
    }
    .secondary-btn button {
      background: #eef3f1 !important;
      color: #23323a !important;
      border: 1px solid var(--border) !important;
    }
    .result-area { min-height: 520px; }
    .result-area img { object-fit: contain; }
    .section-title { font-weight: 600; letter-spacing: 0.2px; }
    """
    with gr.Blocks(title="Microscopy Cell Detection (YOLOv8)", css=css) as demo:
        gr.Markdown(
            "## Microscopy Cell Detection & Counting (YOLOv8)\n"
            "Clinical-style control panel for dataset checks, training, and single-image inference."
        )

        with gr.Tab("Dataset & Labels"):
            with gr.Row():
                with gr.Column(scale=2, min_width=320):
                    gr.Markdown("**Dataset Intake**  \n<span class='muted'>Path-based validation and statistics</span>")
                    dataset_root = gr.Textbox(
                        label="Dataset root path",
                        placeholder="e.g., C:/path/to/datasets/cells OR C:/path/to/data/train",
                    )
                    scan_btn = gr.Button("Scan dataset", elem_classes=["primary-btn"])
                with gr.Column(scale=3, min_width=420):
                    dataset_summary = gr.Textbox(label="Summary", lines=12)
            with gr.Accordion("Raw Details (JSON)", open=False):
                dataset_json = gr.JSON(label="Raw stats (JSON)")

            scan_btn.click(ui_scan_dataset, inputs=[dataset_root], outputs=[dataset_summary, dataset_json])

        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column(scale=2, min_width=320):
                    gr.Markdown("**Training Console**  \n<span class='muted'>Configure model and run training</span>")
                    data_yaml = gr.Textbox(
                        label="Dataset YAML path",
                        placeholder="e.g., C:/path/to/cell.yaml",
                        value=r"C:\Users\hwang\microscopy-yolov8\data\cell.yaml",
                    )
                    model_name = gr.Dropdown(
                        label="YOLOv8 model",
                        choices=[
                            "yolov8n.pt",
                            "yolov8s.pt",
                            "yolov8m.pt",
                            "yolov8l.pt",
                            "yolov8x.pt",
                        ],
                        value="yolov8n.pt",
                    )
                    device = gr.Radio(label="Device", choices=["auto", "cpu", "cuda"], value="auto")
                    train_btn = gr.Button("Start training", elem_classes=["primary-btn"])

                with gr.Column(scale=3, min_width=420):
                    with gr.Accordion("Core Parameters", open=True):
                        with gr.Row():
                            epochs = gr.Number(label="Epochs", value=100, precision=0)
                            batch = gr.Number(label="Batch size", value=16, precision=0)
                            imgsz = gr.Number(label="Image size", value=640, precision=0)
                    with gr.Accordion("Advanced / Subset", open=False):
                        with gr.Row():
                            limit_train = gr.Number(
                                label="Limit train images (0=all)", value=0, precision=0
                            )
                            limit_val = gr.Number(
                                label="Limit val images (0=all)", value=0, precision=0
                            )
                    train_log = gr.Textbox(label="Training log", lines=12)
                    train_status = gr.Textbox(label="Status", lines=1)
                    train_output = gr.JSON(label="Model paths")

            train_btn.click(
                ui_train_model_stream,
                inputs=[data_yaml, model_name, epochs, batch, imgsz, device, limit_train, limit_val],
                outputs=[train_log, train_output, train_status],
            )

        with gr.Tab("Infer & Count"):
            with gr.Row():
                with gr.Column(scale=2, min_width=320):
                    gr.Markdown("**Acquisition & Inference**  \n<span class='muted'>Single-field analysis</span>")
                    weights = gr.Textbox(
                        label="Weights path",
                        placeholder="e.g., runs/detect/cells/weights/best.pt",
                        value=r"C:\Users\hwang\microscopy-yolov8\runs\detect\cells\weights\best.pt",
                    )
                    find_latest = gr.Button("Auto-find latest model", elem_classes=["secondary-btn"])
                    image = gr.Image(
                        label="Upload microscopy image",
                        type="filepath",
                    )
                    infer_btn = gr.Button("Run inference", elem_classes=["primary-btn"])
                    model_status = gr.Textbox(label="Model status", lines=1)

                    with gr.Accordion("Advanced Parameters", open=False):
                        imgsz_i = gr.Number(label="Image size", value=640, precision=0)
                        conf = gr.Slider(
                            label="Confidence", minimum=0.0, maximum=1.0, value=0.25, step=0.01
                        )
                        iou = gr.Slider(
                            label="NMS IoU", minimum=0.0, maximum=1.0, value=0.45, step=0.01
                        )
                        device_i = gr.Radio(label="Device", choices=["auto", "cpu", "cuda"], value="auto")

                with gr.Column(scale=3, min_width=520):
                    output_image = gr.Image(label="Result image", elem_classes=["result-area"])
                    with gr.Row():
                        output_counts = gr.JSON(label="Per-class counts")
                        output_total = gr.Textbox(label="Total cells")

            infer_btn.click(
                ui_infer_count,
                inputs=[weights, image, imgsz_i, conf, iou, device_i],
                outputs=[output_image, output_counts, output_total],
            )
            find_latest.click(
                ui_find_latest_model,
                inputs=[],
                outputs=[weights, model_status],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch()
