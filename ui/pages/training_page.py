from __future__ import annotations

from pathlib import Path
import re
import time

from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFrame,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QPushButton,
    QPlainTextEdit,
    QHBoxLayout,
    QProgressBar,
    QGridLayout,
    QFileDialog,
)

from core import build_val_split, get_run_outputs, scan_dataset_from_yaml, train_yolov8_stream
from core.constants import DEVICE_OPTIONS, MODEL_OPTIONS
from ui.utils import default_yaml_path, get_setting, set_setting


class TrainingWorker(QThread):
    log_line = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        data_yaml: str,
        model_name: str,
        epochs: int,
        batch: int,
        imgsz: int,
        device: str,
        limit_train: int,
        limit_val: int,
        remap_classes: bool,
    ):
        super().__init__()
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device
        self.limit_train = limit_train
        self.limit_val = limit_val
        self.remap_classes = remap_classes

    def run(self) -> None:
        had_error = False
        exit_message = ""
        for line in train_yolov8_stream(
            data_yaml=self.data_yaml,
            model_name=self.model_name,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            limit_train_images=self.limit_train,
            limit_val_images=self.limit_val,
            remap_classes=self.remap_classes,
        ):
            self.log_line.emit(line.rstrip("\n"))
            if line.startswith("[ERROR] Training process exited with code"):
                had_error = True
                exit_message = line.strip()
        outputs = get_run_outputs("runs/detect", "cells")
        if had_error:
            self.failed.emit(exit_message or "Training failed.")
            return
        if not outputs.get("best") and not outputs.get("last"):
            self.failed.emit("Training finished but no weights were found.")
            return
        self.finished.emit(outputs)


class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Training")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #e5e7eb;")
        layout.addWidget(title)

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
            QLineEdit, QComboBox, QSpinBox, QCheckBox {
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
            QPlainTextEdit {
                background: #161b22;
                border: 1px solid #374151;
                border-radius: 6px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
                color: #e5e7eb;
            }
            """
        )

        # ===== Dataset & Model =====
        dataset_group = QGroupBox("Dataset & Model")
        dataset_layout = QFormLayout(dataset_group)

        default_yaml = get_setting("dataset_yaml", str(default_yaml_path()))
        self.data_yaml = QLineEdit(default_yaml)
        self.data_yaml.setReadOnly(True)
        self.browse_yaml_btn = QPushButton("Browse")
        self.browse_yaml_btn.clicked.connect(self.browse_yaml)
        self.model_name = QComboBox()
        self.model_name.addItems(MODEL_OPTIONS)
        self.device = QComboBox()
        self.device.addItems(DEVICE_OPTIONS)

        self.model_name.setCurrentText(get_setting("train_model", "yolov8n.pt"))
        self.device.setCurrentText(get_setting("train_device", "auto"))

        yaml_row = QHBoxLayout()
        yaml_row.addWidget(self.data_yaml, 1)
        yaml_row.addWidget(self.browse_yaml_btn)
        dataset_layout.addRow("Dataset YAML:", yaml_row)
        dataset_layout.addRow("Model:", self.model_name)
        dataset_layout.addRow("Device:", self.device)

        # ===== Training Parameters =====
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_group)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(100)
        self.batch = QSpinBox()
        self.batch.setRange(1, 512)
        self.batch.setValue(16)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(64, 4096)
        self.imgsz.setValue(640)
        self.limit_train = QSpinBox()
        self.limit_train.setRange(0, 1000000)
        self.limit_train.setValue(int(get_setting("limit_train", "0")))
        self.limit_val = QSpinBox()
        self.limit_val.setRange(0, 1000000)
        self.limit_val.setValue(int(get_setting("limit_val", "0")))

        params_layout.addWidget(QLabel("Epochs"), 0, 0)
        params_layout.addWidget(self.epochs, 0, 1)
        params_layout.addWidget(QLabel("Batch"), 0, 2)
        params_layout.addWidget(self.batch, 0, 3)
        params_layout.addWidget(QLabel("Image size"), 1, 0)
        params_layout.addWidget(self.imgsz, 1, 1)
        params_layout.addWidget(QLabel("Limit train images"), 1, 2)
        params_layout.addWidget(self.limit_train, 1, 3)
        params_layout.addWidget(QLabel("Limit val images"), 2, 0)
        params_layout.addWidget(self.limit_val, 2, 1)

        # ===== Actions & Status =====
        actions_group = QGroupBox("Actions & Status")
        actions_layout = QVBoxLayout(actions_group)
        btn_row = QHBoxLayout()

        self.start_btn = QPushButton("Start Training")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_training)
        self.check_btn = QPushButton("Check Dataset")
        self.check_btn.clicked.connect(self.check_dataset)
        self.regen_val_btn = QPushButton("Regenerate Val")
        self.regen_val_btn.clicked.connect(self.regenerate_val)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.check_btn)
        btn_row.addWidget(self.regen_val_btn)
        btn_row.addStretch()

        self.output = QLabel("Status: Idle")
        actions_layout.addLayout(btn_row)
        actions_layout.addWidget(self.output)

        # ===== Logs =====
        logs_group = QGroupBox("Logs")
        logs_layout = QVBoxLayout(logs_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.setVisible(False)
        self.elapsed = QLabel("Elapsed: 00:00")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)

        logs_layout.addWidget(self.elapsed)
        logs_layout.addWidget(self.progress)
        logs_layout.addWidget(self.log)

        layout.addWidget(dataset_group)
        layout.addWidget(params_group)
        layout.addWidget(actions_group)
        layout.addWidget(logs_group)
        layout.addStretch()

        self.worker: TrainingWorker | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_elapsed)
        self.start_time = 0.0

    def start_training(self) -> None:
        self.log.clear()
        if not self.check_dataset(autofix=False):
            return
        self.output.setText("Status: Running...")
        self.start_btn.setEnabled(False)
        self.set_controls_enabled(False)
        self.progress.setVisible(True)
        self.start_time = time.monotonic()
        self.timer.start(500)

        set_setting("dataset_yaml", self.data_yaml.text().strip())
        set_setting("train_model", self.model_name.currentText())
        set_setting("train_device", self.device.currentText())
        set_setting("limit_train", str(self.limit_train.value()))
        set_setting("limit_val", str(self.limit_val.value()))

        self.worker = TrainingWorker(
            data_yaml=self.data_yaml.text().strip(),
            model_name=self.model_name.currentText(),
            epochs=int(self.epochs.value()),
            batch=int(self.batch.value()),
            imgsz=int(self.imgsz.value()),
            device=self.device.currentText(),
            limit_train=int(self.limit_train.value()),
            limit_val=int(self.limit_val.value()),
            remap_classes=False,
        )
        self.worker.log_line.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def check_dataset(self, autofix: bool = False) -> bool:
        try:
            stats = scan_dataset_from_yaml(self.data_yaml.text().strip())
        except Exception as exc:
            self.log.appendPlainText(f"[ERROR] {exc}")
            return False

        split_stats = stats.get("split_stats", {})
        val = split_stats.get("val", {})
        train = split_stats.get("train", {})
        self.log.appendPlainText(
            f"[CHECK] train images={train.get('images', 0)} labels={train.get('labels', 0)} "
            f"| val images={val.get('images', 0)} labels={val.get('labels', 0)}"
        )

        if train.get("images", 0) == 0:
            self.log.appendPlainText("[ERROR] No training images found.")
            return False
        if val.get("images", 0) == 0 or val.get("labels", 0) == 0:
            self.log.appendPlainText("[ERROR] Validation images/labels missing.")
            if autofix:
                return self.regenerate_val()
            return False

        if val.get("missing_labels", 0) > 0:
            self.log.appendPlainText("[WARN] Some val images are missing labels.")
        return True

    def regenerate_val(self) -> bool:
        try:
            try:
                import yaml
            except Exception:
                self.log.appendPlainText(
                    "[ERROR] PyYAML is required. Install with: pip install pyyaml"
                )
                return False
            yaml_path = Path(self.data_yaml.text().strip())
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            root = Path(data.get("path", "")).expanduser()
            stats = build_val_split(str(root), ratio=0.1, seed=42)
            if stats["val_images"] == 0:
                self.log.appendPlainText("[ERROR] Cannot regenerate val: no train images found.")
                return False
            self.log.appendPlainText(
                f"[OK] Regenerated val: {stats['val_images']} images from {stats['train_images']}."
            )
            return True
        except Exception as exc:
            self.log.appendPlainText(f"[ERROR] {exc}")
            return False

    def append_log(self, line: str) -> None:
        clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
        self.log.appendPlainText(clean)

    def on_finished(self, outputs: dict) -> None:
        best_name = Path(outputs.get("best", "")).name
        self.output.setText(f"Status: Completed | best: {best_name}")
        self.start_btn.setEnabled(True)
        self.set_controls_enabled(True)
        self.progress.setVisible(False)
        self.timer.stop()

    def on_failed(self, message: str) -> None:
        self.output.setText(f"Status: Failed | {message}")
        self.start_btn.setEnabled(True)
        self.set_controls_enabled(True)
        self.progress.setVisible(False)
        self.timer.stop()

    def update_elapsed(self) -> None:
        if self.start_time <= 0:
            return
        seconds = int(time.monotonic() - self.start_time)
        mm = seconds // 60
        ss = seconds % 60
        self.elapsed.setText(f"Elapsed: {mm:02d}:{ss:02d}")

    def set_controls_enabled(self, enabled: bool) -> None:
        self.data_yaml.setEnabled(enabled)
        self.model_name.setEnabled(enabled)
        self.device.setEnabled(enabled)
        self.epochs.setEnabled(enabled)
        self.batch.setEnabled(enabled)
        self.imgsz.setEnabled(enabled)
        self.limit_train.setEnabled(enabled)
        self.limit_val.setEnabled(enabled)
        self.check_btn.setEnabled(enabled)
        self.regen_val_btn.setEnabled(enabled)

    def set_dataset_root(self, dataset_root: str) -> None:
        # Align training YAML with dashboard dataset root
        root = Path(dataset_root)
        yaml_path = root / "cell.yaml"
        if yaml_path.exists():
            self.data_yaml.setText(str(yaml_path))
            set_setting("dataset_yaml", str(yaml_path))

    def browse_yaml(self) -> None:
        start_dir = str(Path(self.data_yaml.text().strip()).parent) if self.data_yaml.text().strip() else str(Path(__file__).resolve().parents[2])
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset YAML", start_dir, "YAML Files (*.yaml *.yml)")
        if file_path:
            self.data_yaml.setText(file_path)
            set_setting("dataset_yaml", file_path)
