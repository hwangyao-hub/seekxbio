from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QFormLayout,
    QDoubleSpinBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QCheckBox,
)
from PIL.ImageQt import ImageQt

from core import infer_and_count, infer_batch
from core.constants import CLASS_NAMES_CN, DEVICE_OPTIONS
from ui.annotation_tool import AnnotationDialog
from ui.utils import default_weights_path, find_latest_model, get_setting, project_root, set_setting


class ZoomableImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setRenderHints(self.renderHints() | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._has_image = False
        self._user_zoomed = False
        self._base_fit_scale = 1.0

    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self._has_image = True
        self._user_zoomed = False
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._base_fit_scale = self.transform().m11()

    def wheelEvent(self, event):
        if not self._has_image:
            return
        self._user_zoomed = True
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._has_image and not self._user_zoomed:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._base_fit_scale = self.transform().m11()

    def reset_zoom(self) -> None:
        if not self._has_image:
            return
        self._user_zoomed = False
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom_100(self) -> None:
        if not self._has_image:
            return
        self._user_zoomed = True
        self.resetTransform()
        self.scale(1.0, 1.0)

    def fit_view(self) -> None:
        self.reset_zoom()


class InferenceWorker(QThread):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        weights: str,
        image_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        device: str,
        isolate: bool = True,
        label_mapping: dict[int, str] | None = None,
    ):
        super().__init__()
        self.weights = weights
        self.image_path = image_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.isolate = isolate
        self.label_mapping = label_mapping

    def run(self) -> None:
        try:
            if self.isolate:
                payload = self._run_isolated()
            else:
                vis_img, counts, total, dets = infer_and_count(
                    weights=self.weights,
                    source_image=self.image_path,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    return_dets=True,
                    label_mapping=self.label_mapping,
                )
                payload = (vis_img, counts, total, dets)
            self.finished.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run_isolated(self) -> tuple[object, dict[int, int], int, list[dict]]:
        from PIL import Image

        root = project_root()
        script_path = root / "scripts" / "infer_and_count.py"
        if getattr(sys, "frozen", False) or not script_path.exists():
            vis_img, counts, total, dets = infer_and_count(
                weights=self.weights,
                source_image=self.image_path,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                return_dets=True,
                label_mapping=self.label_mapping,
            )
            return vis_img, counts, total, dets
        save_dir = root / "outputs" / "infer"
        save_dir.mkdir(parents=True, exist_ok=True)
        image_src = Path(self.image_path)
        out_image = save_dir / f"{image_src.stem}_pred.png"
        out_counts = save_dir / f"{image_src.stem}_counts.json"
        out_dets = save_dir / f"{image_src.stem}_dets.json"

        cmd = [
            sys.executable,
            str(script_path),
            "--weights",
            self.weights,
            "--source",
            self.image_path,
            "--imgsz",
            str(self.imgsz),
            "--conf",
            str(self.conf),
            "--iou",
            str(self.iou),
            "--device",
            self.device,
            "--save_dir",
            str(save_dir),
            "--save_dets",
            str(out_dets),
        ]
        if self.label_mapping:
            cmd.append("--use_cn_labels")
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            stderr = (proc.stdout or "") + "\n" + (proc.stderr or "")
            raise RuntimeError(f"Inference failed (code {proc.returncode}).\n{stderr.strip()}")
        if not out_image.exists():
            raise FileNotFoundError(f"Missing output image: {out_image}")
        vis_img = Image.open(out_image).convert("RGB")
        counts_raw = {}
        if out_counts.exists():
            counts_raw = json.loads(out_counts.read_text(encoding="utf-8"))
        counts = {int(k): int(v) for k, v in counts_raw.items()}
        dets: list[dict] = []
        if out_dets.exists():
            dets = json.loads(out_dets.read_text(encoding="utf-8"))
        total = sum(counts.values())
        return vis_img, counts, total, dets


class BatchInferenceWorker(QThread):
    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        weights: str,
        source_dir: str,
        output_dir: str,
        imgsz: int,
        conf: float,
        iou: float,
        device: str,
        label_mapping: dict[int, str] | None = None,
    ):
        super().__init__()
        self.weights = weights
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.label_mapping = label_mapping

    def run(self) -> None:
        try:
            def cb(current, total, filename):
                self.progress.emit(f"{current}/{total} - {filename}")

            results = infer_batch(
                weights=self.weights,
                source_dir=self.source_dir,
                output_dir=self.output_dir,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                label_mapping=self.label_mapping,
                progress_callback=cb,
            )
            summary = (
                f"Total images: {results.get('total_images', 0)}\n"
                f"Total cells: {results.get('total_cells', 0)}\n"
                f"CSV: {results.get('csv_path', '')}"
            )
            self.finished.emit(summary)
        except Exception as exc:
            self.failed.emit(str(exc))


class InferencePage(QWidget):
    model_changed = Signal(str)
    device_changed = Signal(str)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        self.setStyleSheet(
            """
            QWidget { background: #0a0e1a; color: #e5e7eb; }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 8px;
                padding: 10px;
                background: #0d1117;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
                color: #e5e7eb;
                font-weight: 600;
            }
            QLineEdit, QComboBox, QDoubleSpinBox {
                background: #161b22;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 4px;
                color: #e5e7eb;
            }
            QPushButton {
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 6px 10px;
                background: #161b22;
                color: #e5e7eb;
            }
            QPushButton#primary {
                background: #f97316;
                color: #FFFFFF;
                border: 1px solid #f97316;
                font-weight: 600;
            }
            QTableWidget {
                background: #161b22;
                border: 1px solid #374151;
                color: #e5e7eb;
            }
            """
        )

        main = QHBoxLayout()

        # Left panel (controls)
        control_panel = QGroupBox("Acquisition & Parameters")
        control_layout = QVBoxLayout(control_panel)

        form = QFormLayout()
        default_weights = get_setting("weights_path", "")
        if not default_weights:
            latest = default_weights_path()
            if latest is not None:
                default_weights = str(latest)
        self.weights_edit = QLineEdit(default_weights)
        self.weights_browse = QPushButton("Browse")
        self.weights_browse.clicked.connect(self.browse_weights)
        self.image_edit = QLineEdit(get_setting("infer_image", ""))
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_image)

        self.find_latest_btn = QPushButton("Auto-find latest model")
        self.find_latest_btn.clicked.connect(self.auto_find_latest)

        image_row = QHBoxLayout()
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(browse_btn)

        weights_row = QHBoxLayout()
        weights_row.addWidget(self.weights_edit, 1)
        weights_row.addWidget(self.weights_browse)
        form.addRow("Weights:", weights_row)
        form.addRow("", self.find_latest_btn)
        form.addRow("Image:", image_row)

        self.imgsz = QDoubleSpinBox()
        self.imgsz.setRange(64, 4096)
        self.imgsz.setValue(640)
        self.imgsz.setDecimals(0)

        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.0, 1.0)
        self.conf.setSingleStep(0.01)
        self.conf.setValue(0.25)

        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.0, 1.0)
        self.iou.setSingleStep(0.01)
        self.iou.setValue(0.45)

        self.device = QComboBox()
        self.device.addItems(DEVICE_OPTIONS)
        self.device.setCurrentText(get_setting("infer_device", "auto"))
        self.device.currentTextChanged.connect(self.on_device_changed)

        form.addRow("Image size:", self.imgsz)
        form.addRow("Conf:", self.conf)
        form.addRow("IoU:", self.iou)
        form.addRow("Device:", self.device)

        self.run_btn = QPushButton("Run Inference")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run_inference)
        self.isolate_infer = QCheckBox("Isolate inference (safer)")
        self.isolate_infer.setChecked(get_setting("infer_isolate", "1") == "1")

        self.counts_table = QTableWidget(0, 2)
        self.counts_table.setHorizontalHeaderLabels(["Class", "Count"])
        self.counts_table.horizontalHeader().setStretchLastSection(True)
        self.total_label = QLabel("Total cells: 0")
        self.status_label = QLabel("Status: Idle")

        control_layout.addLayout(form)
        control_layout.addWidget(self.isolate_infer)
        control_layout.addWidget(self.run_btn)
        control_layout.addWidget(self.counts_table)
        control_layout.addWidget(self.total_label)
        control_layout.addWidget(self.status_label)

        self.batch_group = QGroupBox("Batch Inference")
        batch_layout = QVBoxLayout(self.batch_group)
        self.batch_input = QLineEdit()
        self.batch_output = QLineEdit(str(project_root() / "outputs" / "batch_infer"))
        self.batch_run_btn = QPushButton("Run Batch Inference")
        self.batch_progress = QLabel("Progress: Idle")
        self.batch_result = QLabel("")

        batch_layout.addWidget(QLabel("Input dir:"))
        batch_layout.addWidget(self.batch_input)
        batch_layout.addWidget(QLabel("Output dir:"))
        batch_layout.addWidget(self.batch_output)
        batch_layout.addWidget(self.batch_run_btn)
        batch_layout.addWidget(self.batch_progress)
        batch_layout.addWidget(self.batch_result)

        self.batch_run_btn.clicked.connect(self.run_batch_inference)

        control_layout.addWidget(self.batch_group)

        # Right panel (results)
        result_panel = QGroupBox("Results")
        result_layout = QVBoxLayout(result_panel)

        self.image_view = ZoomableImageView()
        self.image_view.setMinimumHeight(420)

        self.annotate_btn = QPushButton("Open Annotator")
        self.annotate_btn.setEnabled(False)
        self.annotate_btn.clicked.connect(self.open_annotator)

        zoom_row = QHBoxLayout()
        self.zoom_reset_btn = QPushButton("Reset Zoom")
        self.zoom_100_btn = QPushButton("100%")
        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_reset_btn.clicked.connect(self.image_view.reset_zoom)
        self.zoom_100_btn.clicked.connect(self.image_view.zoom_100)
        self.zoom_fit_btn.clicked.connect(self.image_view.fit_view)
        zoom_row.addWidget(self.annotate_btn)
        zoom_row.addWidget(self.zoom_reset_btn)
        zoom_row.addWidget(self.zoom_100_btn)
        zoom_row.addWidget(self.zoom_fit_btn)
        zoom_row.addStretch()

        result_layout.addWidget(self.image_view, 1)
        result_layout.addLayout(zoom_row)

        main.addWidget(control_panel, 1)
        main.addWidget(result_panel, 3)
        layout.addLayout(main, 1)

        self.worker: InferenceWorker | None = None
        self.batch_worker: BatchInferenceWorker | None = None
        self.last_image_path: str | None = None
        self.last_detections: list[dict] = []

    def browse_image(self) -> None:
        start_dir = get_setting("infer_browse_dir", "")
        if not start_dir:
            start_dir = str(project_root())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if file_path:
            self.image_edit.setText(file_path)
            set_setting("infer_browse_dir", str(Path(file_path).parent))

    def browse_weights(self) -> None:
        start_dir = str(Path(self.weights_edit.text().strip()).parent) if self.weights_edit.text().strip() else str(project_root())
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Weights", start_dir, "Weights (*.pt)")
        if file_path:
            self.weights_edit.setText(file_path)
            set_setting("weights_path", file_path)

    def auto_find_latest(self) -> None:
        latest = find_latest_model(project_root() / "runs" / "detect")
        if latest is not None:
            self.weights_edit.setText(str(latest))
            self.model_changed.emit(str(latest))

    def on_device_changed(self, device: str) -> None:
        set_setting("infer_device", device)
        self.device_changed.emit(device)

    def run_inference(self) -> None:
        weights = self.weights_edit.text().strip()
        image_path = self.image_edit.text().strip()
        if not weights or not image_path:
            self.total_label.setText("Total cells: 0 (missing inputs)")
            return

        self.model_changed.emit(weights)
        set_setting("weights_path", weights)
        set_setting("infer_image", image_path)
        set_setting("infer_isolate", "1" if self.isolate_infer.isChecked() else "0")

        self.run_btn.setEnabled(False)
        self.status_label.setText("Status: Running...")
        self.worker = InferenceWorker(
            weights=weights,
            image_path=image_path,
            imgsz=int(self.imgsz.value()),
            conf=float(self.conf.value()),
            iou=float(self.iou.value()),
            device=self.device.currentText(),
            isolate=self.isolate_infer.isChecked(),
            label_mapping=CLASS_NAMES_CN,
        )
        self.worker.finished.connect(self.on_inference_done)
        self.worker.failed.connect(self.on_inference_failed)
        self.worker.start()

    def on_inference_done(self, payload) -> None:
        vis_img, counts, total, dets = payload
        qt_img = ImageQt(vis_img)
        pixmap = QPixmap.fromImage(qt_img)
        self.image_view.set_image(pixmap)

        self.counts_table.setRowCount(0)
        class_names = []
        label_mapping = CLASS_NAMES_CN
        if not counts and dets:
            from collections import Counter
            counts = {int(k): v for k, v in Counter(int(d["class_id"]) for d in dets).items()}
        for cls_id, count in counts.items():
            row = self.counts_table.rowCount()
            self.counts_table.insertRow(row)
            display_name = str(cls_id)
            if label_mapping and cls_id in label_mapping:
                display_name = label_mapping[cls_id]
            elif class_names and cls_id < len(class_names):
                display_name = class_names[cls_id]
            self.counts_table.setItem(row, 0, QTableWidgetItem(display_name))
            self.counts_table.setItem(row, 1, QTableWidgetItem(str(count)))
        self.total_label.setText(f"Total cells: {total}")
        self.status_label.setText("Status: Completed")

        self.last_image_path = self.image_edit.text().strip()
        self.last_detections = dets
        self.annotate_btn.setEnabled(bool(self.last_image_path))
        self.run_btn.setEnabled(True)

    def on_inference_failed(self, message: str) -> None:
        self.total_label.setText(f"Error: {message}")
        self.status_label.setText("Status: Failed")
        self.run_btn.setEnabled(True)

    def open_annotator(self) -> None:
        if not getattr(self, "last_image_path", ""):
            self.status_label.setText("Status: Missing image path")
            return
        dets = getattr(self, "last_detections", [])
        dialog = AnnotationDialog(self.last_image_path, dets, self)
        dialog.exec()

    def run_batch_inference(self) -> None:
        weights = self.weights_edit.text().strip()
        source_dir = self.batch_input.text().strip()
        output_dir = self.batch_output.text().strip()
        if not weights or not source_dir or not output_dir:
            self.batch_progress.setText("Progress: missing inputs")
            return
        self.batch_progress.setText("Progress: running...")
        self.batch_result.setText("")
        self.batch_run_btn.setEnabled(False)
        self.batch_worker = BatchInferenceWorker(
            weights=weights,
            source_dir=source_dir,
            output_dir=output_dir,
            imgsz=int(self.imgsz.value()),
            conf=float(self.conf.value()),
            iou=float(self.iou.value()),
            device=self.device.currentText(),
            label_mapping=CLASS_NAMES_CN,
        )
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.failed.connect(self.on_batch_failed)
        self.batch_worker.start()

    def on_batch_progress(self, message: str) -> None:
        self.batch_progress.setText(f"Progress: {message}")

    def on_batch_finished(self, summary: str) -> None:
        self.batch_progress.setText("Progress: completed")
        self.batch_result.setText(summary)
        self.batch_run_btn.setEnabled(True)

    def on_batch_failed(self, message: str) -> None:
        self.batch_progress.setText(f"Progress: failed ({message})")
        self.batch_run_btn.setEnabled(True)

    def set_dataset_root(self, dataset_root: str) -> None:
        # Align weights path with dashboard dataset root's parent (project root)
        root = Path(dataset_root)
        project_root_guess = root.parent if root.name.lower() == "data" else root
        weights = project_root_guess / "runs" / "detect" / "cells" / "weights" / "best.pt"
        if weights.exists():
            self.weights_edit.setText(str(weights))
            set_setting("weights_path", str(weights))
