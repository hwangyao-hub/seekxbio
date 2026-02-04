from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QPushButton

from ui.utils import read_version


class TopBar(QWidget):
    tab_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(96)
        self.setStyleSheet(
            """
            QWidget {
                background: #0d1117;
                color: #e5e7eb;
            }
            QLabel { font-weight: 500; }
            QPushButton {
                background: transparent;
                border: none;
                color: #9ca3af;
                padding: 8px 12px;
                font-size: 13px;
            }
            QPushButton:checked {
                color: #60a5fa;
                border-bottom: 2px solid #3b82f6;
                font-weight: 600;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 10, 16, 0)
        root.setSpacing(8)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self.app_name = QLabel("Microscopy AI - 细胞检测系统")
        self.subtitle = QLabel("基于 YOLOv8 的显微镜血细胞检测、分类与计数")
        self.subtitle.setStyleSheet("color: #9ca3af; font-size: 12px;")
        title_box.addWidget(self.app_name)
        title_box.addWidget(self.subtitle)

        header.addLayout(title_box)
        header.addStretch()

        self.version_info = QLabel(f"v{read_version()}")
        self.version_info.setStyleSheet("color: #9ca3af; font-size: 12px;")
        self.model_info = QLabel("Model: -")
        self.model_info.setStyleSheet("color: #9ca3af; font-size: 12px;")
        self.device_info = QLabel("Device: CPU")
        self.device_info.setStyleSheet("color: #9ca3af; font-size: 12px;")

        header.addWidget(self.version_info)
        header.addSpacing(16)
        header.addWidget(self.model_info)
        header.addSpacing(12)
        header.addWidget(self.device_info)

        tabs = QHBoxLayout()
        tabs.setSpacing(4)

        self.buttons: list[QPushButton] = []
        self.tab_names = ["数据仪表板", "模型训练", "推理检测", "设置"]
        for idx, name in enumerate(self.tab_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=idx: self.switch_tab(i))
            tabs.addWidget(btn)
            self.buttons.append(btn)
        tabs.addStretch()
        self.buttons[0].setChecked(True)

        root.addLayout(header)
        root.addLayout(tabs)

    def switch_tab(self, index: int) -> None:
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)
        self.tab_changed.emit(index)

    def set_active_tab(self, index: int) -> None:
        self.switch_tab(index)

    def set_model(self, model_path: str) -> None:
        if not model_path or model_path == "-":
            name = "-"
        else:
            name = Path(model_path).name
        self.model_info.setText(f"Model: {name}")

    def set_device(self, device: str) -> None:
        device_label = device.upper() if device else "CPU"
        self.device_info.setText(f"Device: {device_label}")
