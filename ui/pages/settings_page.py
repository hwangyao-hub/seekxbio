from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
)

from core import load_class_mapping_rows, save_class_mapping_rows
from core.utils import resolve_device
from ui.utils import project_root


class SettingsPage(QWidget):
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
                padding: 12px;
                background: #0d1117;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
                color: #e5e7eb;
                font-weight: 600;
            }
            QLineEdit, QTableWidget {
                background: #161b22;
                border: 1px solid #374151;
                border-radius: 6px;
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
                color: #ffffff;
                border: 1px solid #f97316;
                font-weight: 600;
            }
            QLabel#muted {
                color: #9ca3af;
            }
            """
        )

        info_group = QGroupBox("系统信息")
        info_layout = QVBoxLayout(info_group)
        device_label = QLabel(f"PyTorch: {resolve_device('auto')}")
        device_label.setObjectName("muted")
        info_layout.addWidget(device_label)

        mapping_group = QGroupBox("类别名称映射")
        mapping_layout = QVBoxLayout(mapping_group)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["ID", "English", "中文"])
        self.table.horizontalHeader().setStretchLastSection(True)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("添加行")
        self.remove_btn = QPushButton("删除选中")
        self.save_btn = QPushButton("保存映射")
        self.save_btn.setObjectName("primary")
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.save_btn)

        status_row = QHBoxLayout()
        self.status = QLineEdit()
        self.status.setReadOnly(True)
        status_row.addWidget(self.status)

        mapping_layout.addWidget(self.table)
        mapping_layout.addLayout(btn_row)
        mapping_layout.addLayout(status_row)

        layout.addWidget(info_group)
        layout.addWidget(mapping_group)
        layout.addStretch()

        self.add_btn.clicked.connect(self.add_row)
        self.remove_btn.clicked.connect(self.remove_row)
        self.save_btn.clicked.connect(self.save_rows)

        self.load_rows()

    def load_rows(self) -> None:
        rows = load_class_mapping_rows(project_root())
        self.table.setRowCount(0)
        for row in rows:
            ridx = self.table.rowCount()
            self.table.insertRow(ridx)
            self.table.setItem(ridx, 0, QTableWidgetItem(str(row.get("id", ""))))
            self.table.setItem(ridx, 1, QTableWidgetItem(str(row.get("en", ""))))
            self.table.setItem(ridx, 2, QTableWidgetItem(str(row.get("cn", ""))))

    def add_row(self) -> None:
        row_count = self.table.rowCount()
        next_id = row_count
        existing = set()
        for i in range(row_count):
            item = self.table.item(i, 0)
            if item is None:
                continue
            try:
                existing.add(int(item.text()))
            except Exception:
                continue
        while next_id in existing:
            next_id += 1
        self.table.insertRow(row_count)
        self.table.setItem(row_count, 0, QTableWidgetItem(str(next_id)))
        self.table.setItem(row_count, 1, QTableWidgetItem(""))
        self.table.setItem(row_count, 2, QTableWidgetItem(""))

    def remove_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def save_rows(self) -> None:
        rows: list[dict[str, object]] = []
        for i in range(self.table.rowCount()):
            id_item = self.table.item(i, 0)
            if id_item is None:
                continue
            try:
                cid = int(id_item.text())
            except Exception:
                continue
            en = self._text_at(i, 1)
            cn = self._text_at(i, 2)
            rows.append({"id": cid, "en": en, "cn": cn})
        if not rows:
            self.status.setText("未发现可保存的数据。")
            return
        path = save_class_mapping_rows(project_root(), rows)
        self.status.setText(f"已保存: {path}")

    def _text_at(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        if item is None:
            return ""
        return item.text().strip()
