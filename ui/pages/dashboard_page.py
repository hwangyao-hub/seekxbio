from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QFileDialog,
)

from core import scan_dataset
from ui.utils import get_setting, set_setting


class DashboardPage(QWidget):
    dataset_root_changed = Signal(str)
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Dashboard")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #111827;")
        layout.addWidget(title)

        self.setStyleSheet(
            """
            QWidget { background: #F8FAFC; color: #111827; }
            QGroupBox {
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                margin-top: 8px;
                padding: 10px;
                background: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
                color: #111827;
                font-weight: 600;
            }
            QLineEdit, QTextEdit {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                padding: 6px 10px;
                background: #FFFFFF;
            }
            """
        )

        card = QGroupBox("Dataset Overview")
        card_layout = QVBoxLayout(card)

        input_row = QHBoxLayout()
        default_root = get_setting(
            "dataset_root", str(Path(__file__).resolve().parents[2] / "data")
        )
        self.dataset_path = QLineEdit(default_root)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_dataset)
        self.scan_btn = QPushButton("Scan Dataset")
        self.scan_btn.clicked.connect(self.scan_dataset)

        input_row.addWidget(QLabel("Dataset Root:"))
        input_row.addWidget(self.dataset_path, 1)
        input_row.addWidget(self.browse_btn)
        input_row.addWidget(self.scan_btn)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(140)

        card_layout.addLayout(input_row)
        card_layout.addWidget(self.summary)

        layout.addWidget(card)
        layout.addStretch()

    def scan_dataset(self) -> None:
        try:
            dataset_root = self.dataset_path.text().strip()
            stats = scan_dataset(dataset_root)
            set_setting("dataset_root", dataset_root)
            self.dataset_root_changed.emit(dataset_root)
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
            for split, s in stats.get("split_stats", {}).items():
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
            self.summary.setPlainText("\n".join(lines))
        except Exception as exc:
            self.summary.setPlainText(f"Error: {exc}")

    def browse_dataset(self) -> None:
        start_dir = self.dataset_path.text().strip()
        if not start_dir:
            start_dir = str(Path(__file__).resolve().parents[2])
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Root", start_dir)
        if folder:
            self.dataset_path.setText(folder)
