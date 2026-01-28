from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal


class SideBar(QWidget):
    page_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(160)
        self.setStyleSheet(
            """
            QWidget { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
            QPushButton {
                text-align: left;
                padding: 8px 10px;
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                background: #FFFFFF;
                color: #111827;
            }
            QPushButton:checked {
                background: #EFF6FF;
                border: 1px solid #2563EB;
                color: #1D4ED8;
                font-weight: 600;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.buttons = []

        pages = ["Dashboard", "Inference", "Training"]
        for idx, name in enumerate(pages):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=idx: self.switch_page(i))
            layout.addWidget(btn)
            self.buttons.append(btn)

        self.buttons[0].setChecked(True)
        layout.addStretch()

    def switch_page(self, index):
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)
        self.page_changed.emit(index)
