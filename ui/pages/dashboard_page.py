from pathlib import Path
import io

from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap
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

from core import scan_dataset, save_dataset_report
from core.constants import CLASS_NAMES_CN
from ui.utils import get_setting, set_setting


class DashboardPage(QWidget):
    dataset_root_changed = Signal(str)
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Dashboard")
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
            QLineEdit, QTextEdit {
                background: #161b22;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 4px;
            }
            QPushButton {
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 6px 10px;
                background: #161b22;
                color: #e5e7eb;
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

        self.chart_label = QLabel("Class distribution chart will appear here after scan.")
        self.chart_label.setMinimumHeight(220)
        self.chart_label.setStyleSheet("background: #0d1117; border: 1px solid #30363d;")
        self.chart_label.setScaledContents(True)

        card_layout.addLayout(input_row)
        card_layout.addWidget(self.summary)
        card_layout.addWidget(self.chart_label)

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
                for k, v in sorted(class_counts.items(), key=lambda kv: int(kv[0])):
                    class_id = int(k)
                    cn_name = CLASS_NAMES_CN.get(class_id, f"Class {class_id}")
                    lines.append(f"- {cn_name} (ID {class_id}): {v}")
            else:
                lines.append("- none")
            report_path = save_dataset_report(
                stats,
                str(Path(__file__).resolve().parents[2] / "outputs" / "reports"),
                class_name_map=CLASS_NAMES_CN,
            )
            lines.append("")
            lines.append(f"Report saved: {report_path}")
            self.summary.setPlainText("\n".join(lines))
            self._update_chart(stats)
        except Exception as exc:
            self.summary.setPlainText(f"Error: {exc}")
            self.chart_label.setText("Chart unavailable.")

    def browse_dataset(self) -> None:
        start_dir = self.dataset_path.text().strip()
        if not start_dir:
            start_dir = str(Path(__file__).resolve().parents[2])
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Root", start_dir)
        if folder:
            self.dataset_path.setText(folder)

    def _update_chart(self, stats: dict) -> None:
        class_counts = stats.get("class_counts", {})
        if not class_counts:
            self.chart_label.setText("No class counts available.")
            self.chart_label.setPixmap(QPixmap())
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            items = sorted(class_counts.items(), key=lambda x: int(x[0]))
            ids = [int(k) for k, _ in items]
            counts = [v for _, v in items]
            labels = [f"{CLASS_NAMES_CN.get(i, str(i))}\n(ID:{i})" for i in ids]

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#0d1117")
            bars = ax.bar(
                range(len(ids)),
                counts,
                color="#3B82F6",
                edgecolor="#1D4ED8",
                linewidth=1.2,
            )
            ax.set_xticks(range(len(ids)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#9ca3af")
            ax.set_xlabel("Cell Class", fontsize=11, color="#9ca3af")
            ax.set_ylabel("Count", fontsize=11, color="#9ca3af")
            ax.set_title("Dataset Class Distribution", fontsize=12, fontweight="bold", color="#e5e7eb")
            ax.grid(axis="y", alpha=0.3, linestyle="--", color="#374151")
            ax.tick_params(colors="#9ca3af")
            ax.set_facecolor("#0d1117")
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#e5e7eb",
                )
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue(), "PNG")
            self.chart_label.setPixmap(pixmap)
        except Exception as exc:
            self.chart_label.setText(f"Chart generation failed: {exc}")
