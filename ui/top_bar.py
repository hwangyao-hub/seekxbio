from pathlib import Path

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel


class TopBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(44)
        self.setStyleSheet(
            """
            QWidget {
                background: #FFFFFF;
                border-bottom: 1px solid #E5E7EB;
                color: #111827;
            }
            QLabel { font-weight: 500; }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)

        self.app_name = QLabel("Microscopy AI")
        self.model_info = QLabel("Model: -")
        self.device_info = QLabel("Device: CPU")

        layout.addWidget(self.app_name)
        layout.addStretch()
        layout.addWidget(self.model_info)
        layout.addSpacing(20)
        layout.addWidget(self.device_info)

    def set_model(self, model_path: str) -> None:
        if not model_path or model_path == "-":
            name = "-"
        else:
            name = Path(model_path).name
        self.model_info.setText(f"Model: {name}")

    def set_device(self, device: str) -> None:
        device_label = device.upper() if device else "CPU"
        self.device_info.setText(f"Device: {device_label}")
