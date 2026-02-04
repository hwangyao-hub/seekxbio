from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPixmap, QPen, QColor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsItem,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QComboBox,
    QLineEdit,
)
from PySide6.QtGui import QShortcut, QKeySequence

from core import image_to_label_path, load_class_mapping_rows, rows_to_maps
from ui.utils import project_root


@dataclass
class Annotation:
    ann_id: int
    class_id: int
    label: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float | None = None


class AnnotationRectItem(QGraphicsRectItem):
    def __init__(
        self,
        ann_id: int,
        class_id: int,
        label: str,
        rect: QRectF,
        on_changed: Callable[["AnnotationRectItem"], None] | None = None,
    ):
        super().__init__(rect)
        self.ann_id = ann_id
        self.class_id = class_id
        self.label = label
        self.on_changed = on_changed
        pen = QPen(QColor("#3b82f6"))
        pen.setWidth(2)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._resizing = False
        self._resize_handle: str | None = None
        self._start_rect: QRectF | None = None
        self._start_pos: QPointF | None = None
        self._handles: dict[str, QGraphicsRectItem] = {}
        self._handle_size = 8
        self._init_handles()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.on_changed:
            self.on_changed(self)

    def bbox(self) -> tuple[float, float, float, float]:
        rect = self.sceneBoundingRect()
        return rect.left(), rect.top(), rect.right(), rect.bottom()

    def _init_handles(self) -> None:
        for key in ("tl", "tr", "bl", "br"):
            handle = QGraphicsRectItem(self)
            handle.setRect(0, 0, self._handle_size, self._handle_size)
            handle.setPen(QPen(QColor("#f97316")))
            handle.setBrush(QColor("#f97316"))
            handle.setFlag(QGraphicsItem.ItemIsSelectable, False)
            handle.setZValue(2)
            self._handles[key] = handle
        self._update_handles()

    def _update_handles(self) -> None:
        r = self.rect()
        hs = self._handle_size / 2.0
        self._handles["tl"].setPos(r.left() - hs, r.top() - hs)
        self._handles["tr"].setPos(r.right() - hs, r.top() - hs)
        self._handles["bl"].setPos(r.left() - hs, r.bottom() - hs)
        self._handles["br"].setPos(r.right() - hs, r.bottom() - hs)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            for key, handle in self._handles.items():
                if handle.contains(handle.mapFromScene(event.scenePos())):
                    self._resizing = True
                    self._resize_handle = key
                    self._start_rect = self.rect()
                    self._start_pos = event.scenePos()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing and self._start_rect and self._start_pos and self._resize_handle:
            dx = event.scenePos().x() - self._start_pos.x()
            dy = event.scenePos().y() - self._start_pos.y()
            r = QRectF(self._start_rect)
            if self._resize_handle == "tl":
                r.setTopLeft(r.topLeft() + QPointF(dx, dy))
            elif self._resize_handle == "tr":
                r.setTopRight(r.topRight() + QPointF(dx, dy))
            elif self._resize_handle == "bl":
                r.setBottomLeft(r.bottomLeft() + QPointF(dx, dy))
            elif self._resize_handle == "br":
                r.setBottomRight(r.bottomRight() + QPointF(dx, dy))
            if r.width() < 5:
                r.setWidth(5)
            if r.height() < 5:
                r.setHeight(5)
            self.setRect(r.normalized())
            self._update_handles()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self._resize_handle = None
            self._start_rect = None
            self._start_pos = None
        self._update_handles()
        super().mouseReleaseEvent(event)


class AnnotationView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self._scene = scene
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._adding = False
        self._start_pos: QPointF | None = None
        self._temp_rect: QGraphicsRectItem | None = None
        self.on_new_rect: Callable[[QRectF], None] | None = None

    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_add_mode(self, enabled: bool) -> None:
        self._adding = enabled
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if self._adding and event.button() == Qt.LeftButton:
            self._start_pos = self.mapToScene(event.position().toPoint())
            self._temp_rect = QGraphicsRectItem()
            pen = QPen(QColor("#f97316"))
            pen.setWidth(2)
            self._temp_rect.setPen(pen)
            self._scene.addItem(self._temp_rect)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._adding and self._temp_rect and self._start_pos:
            end_pos = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._start_pos, end_pos).normalized()
            self._temp_rect.setRect(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._adding and self._temp_rect and self._start_pos:
            rect = self._temp_rect.rect()
            self._scene.removeItem(self._temp_rect)
            self._temp_rect = None
            self._start_pos = None
            if rect.width() > 5 and rect.height() > 5 and self.on_new_rect:
                self.on_new_rect(rect)
            return
        super().mouseReleaseEvent(event)


class AnnotationDialog(QDialog):
    def __init__(
        self,
        image_path: str,
        detections: list[dict],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("标注复检")
        self.resize(1200, 800)
        self.setStyleSheet(
            """
            QDialog { background: #0a0e1a; color: #e5e7eb; }
            QLabel { color: #e5e7eb; }
            QTableWidget { background: #161b22; border: 1px solid #374151; color: #e5e7eb; }
            QLineEdit { background: #161b22; border: 1px solid #374151; color: #e5e7eb; }
            QPushButton { background: #161b22; border: 1px solid #374151; color: #e5e7eb; padding: 6px 10px; border-radius: 6px; }
            """
        )

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        self.scene = QGraphicsScene(self)
        self.view = AnnotationView(self.scene)
        self.view.setMinimumWidth(720)
        self.view.setStyleSheet("background: #0d1117; border: 1px solid #30363d;")
        root.addWidget(self.view, 3)

        right = QVBoxLayout()
        right.setSpacing(8)
        root.addLayout(right, 1)

        self.class_map = rows_to_maps(load_class_mapping_rows(project_root()))[1]
        self.class_choices = sorted(self.class_map.items(), key=lambda kv: kv[0])
        if not self.class_choices:
            self.class_choices = [(0, "class 0")]

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("类别:"))
        self.class_select = QComboBox()
        for cid, name in self.class_choices:
            self.class_select.addItem(f"{cid}: {name}", userData=cid)
        class_row.addWidget(self.class_select)
        right.addLayout(class_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Class", "Conf", "BBox"])
        self.table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("添加框")
        self.delete_btn = QPushButton("删除框")
        self.undo_btn = QPushButton("↩ 撤销")
        self.redo_btn = QPushButton("↪ 重做")
        self.save_btn = QPushButton("保存标签")
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.undo_btn)
        btn_row.addWidget(self.redo_btn)
        btn_row.addWidget(self.save_btn)
        right.addLayout(btn_row)

        self.status = QLineEdit()
        self.status.setReadOnly(True)
        right.addWidget(self.status)

        self.add_btn.clicked.connect(self.enable_add_mode)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.save_btn.clicked.connect(self.save_labels)
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        self.table.itemSelectionChanged.connect(self.on_table_select)
        self.class_select.currentIndexChanged.connect(self.on_class_change)
        self.scene.selectionChanged.connect(self.on_scene_select)

        self.image_path = image_path
        self.image_size = (0, 0)
        self._next_id = 0
        self.items: dict[int, AnnotationRectItem] = {}
        self._history: list[list[dict]] = []
        self._redo_stack: list[list[dict]] = []
        self._max_history = 50
        self._clipboard: list[dict] | None = None
        self._restoring = False

        self.load_image()
        self.load_detections(detections)
        self._save_state()
        self._setup_shortcuts()

    def load_image(self) -> None:
        pixmap = QPixmap(self.image_path)
        self.image_size = (pixmap.width(), pixmap.height())
        self.view.set_image(pixmap)

    def load_detections(self, detections: list[dict]) -> None:
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            cid = int(det.get("class_id", 0))
            label = self.class_map.get(cid, str(cid))
            conf = det.get("confidence")
            rect = QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            self.add_annotation(rect, cid, label, conf)

    def add_annotation(
        self,
        rect: QRectF,
        class_id: int,
        label: str,
        confidence: float | None = None,
    ) -> None:
        ann_id = self._next_id
        self._next_id += 1
        item = AnnotationRectItem(
            ann_id=ann_id,
            class_id=class_id,
            label=label,
            rect=rect,
            on_changed=self.on_item_changed,
        )
        self.scene.addItem(item)
        self.items[ann_id] = item
        self.add_table_row(item, confidence)
        self._save_state()

    def _add_annotation_from_dict(self, ann: dict) -> None:
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            return
        class_id = int(ann.get("class_id", 0))
        label = self.class_map.get(class_id, str(class_id))
        rect = QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        self.add_annotation(rect, class_id, label, ann.get("confidence"))

    def add_table_row(self, item: AnnotationRectItem, confidence: float | None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(item.ann_id)))
        self.table.setItem(row, 1, QTableWidgetItem(item.label))
        self.table.setItem(row, 2, QTableWidgetItem("" if confidence is None else f"{confidence:.2f}"))
        bbox_str = self._bbox_str(item)
        self.table.setItem(row, 3, QTableWidgetItem(bbox_str))

    def _bbox_str(self, item: AnnotationRectItem) -> str:
        x1, y1, x2, y2 = item.bbox()
        return f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"

    def enable_add_mode(self) -> None:
        self.view.set_add_mode(True)
        self.view.on_new_rect = self.on_new_rect
        self.status.setText("拖拽绘制新框，完成后自动添加。")

    def on_new_rect(self, rect: QRectF) -> None:
        class_id = int(self.class_select.currentData())
        label = self.class_map.get(class_id, str(class_id))
        self.add_annotation(rect, class_id, label, None)
        self.view.set_add_mode(False)
        self.status.setText("已添加新框。")

    def on_item_changed(self, item: AnnotationRectItem) -> None:
        self.refresh_table()
        self._save_state()

    def refresh_table(self) -> None:
        self.table.setRowCount(0)
        for ann_id in sorted(self.items.keys()):
            item = self.items[ann_id]
            self.add_table_row(item, None)
            item._update_handles()

    def delete_selected(self) -> None:
        selected = self.scene.selectedItems()
        if not selected:
            return
        for item in selected:
            if isinstance(item, AnnotationRectItem):
                ann_id = item.ann_id
                self.scene.removeItem(item)
                self.items.pop(ann_id, None)
        self.refresh_table()
        self._save_state()

    def on_table_select(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        ann_id_item = self.table.item(row, 0)
        if ann_id_item is None:
            return
        try:
            ann_id = int(ann_id_item.text())
        except Exception:
            return
        item = self.items.get(ann_id)
        if item:
            item.setSelected(True)
            self.class_select.setCurrentIndex(
                self.class_select.findData(item.class_id)
            )

    def on_scene_select(self) -> None:
        selected = self.scene.selectedItems()
        if not selected:
            return
        item = selected[0]
        if isinstance(item, AnnotationRectItem):
            self.class_select.setCurrentIndex(
                self.class_select.findData(item.class_id)
            )

    def on_class_change(self) -> None:
        selected = self.scene.selectedItems()
        if not selected:
            return
        item = selected[0]
        if isinstance(item, AnnotationRectItem):
            class_id = int(self.class_select.currentData())
            item.class_id = class_id
            item.label = self.class_map.get(class_id, str(class_id))
            self.refresh_table()
            self._save_state()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_selected()
            event.accept()
            return
        if event.key() == Qt.Key_N:
            self.enable_add_mode()
            event.accept()
            return
        if Qt.Key_0 <= event.key() <= Qt.Key_9:
            digit = event.key() - Qt.Key_0
            idx = self.class_select.findData(digit)
            if idx >= 0:
                self.class_select.setCurrentIndex(idx)
                self.on_class_change()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self.view.set_add_mode(False)
            self.status.setText("已退出新增模式。")
            event.accept()
            return
        super().keyPressEvent(event)

    def _setup_shortcuts(self) -> None:
        undo_shortcut = QShortcut(QKeySequence.Undo, self)
        undo_shortcut.activated.connect(self.undo)
        redo_shortcut = QShortcut(QKeySequence.Redo, self)
        redo_shortcut.activated.connect(self.redo)
        copy_shortcut = QShortcut(QKeySequence.Copy, self)
        copy_shortcut.activated.connect(self.copy_selected)
        paste_shortcut = QShortcut(QKeySequence.Paste, self)
        paste_shortcut.activated.connect(self.paste)

    def _save_state(self) -> None:
        if self._restoring:
            return
        state = self._get_current_annotations()
        if self._history and self._history[-1] == state:
            return
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._redo_stack.clear()

    def _get_current_annotations(self) -> list[dict]:
        annotations = []
        for item in self.items.values():
            x1, y1, x2, y2 = item.bbox()
            annotations.append(
                {
                    "class_id": item.class_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": None,
                }
            )
        return annotations

    def _restore_state(self, state: list[dict]) -> None:
        self._restoring = True
        for item in list(self.items.values()):
            self.scene.removeItem(item)
        self.items.clear()
        self.table.setRowCount(0)
        self._next_id = 0
        for ann in state:
            self._add_annotation_from_dict(ann)
        self.refresh_table()
        self._restoring = False

    def undo(self) -> None:
        if len(self._history) > 1:
            current = self._history.pop()
            self._redo_stack.append(current)
            self._restore_state(self._history[-1])

    def redo(self) -> None:
        if self._redo_stack:
            state = self._redo_stack.pop()
            self._history.append(state)
            self._restore_state(state)

    def copy_selected(self) -> None:
        selected = self.scene.selectedItems()
        if not selected:
            return
        item = selected[0]
        if isinstance(item, AnnotationRectItem):
            x1, y1, x2, y2 = item.bbox()
            self._clipboard = [
                {
                    "class_id": item.class_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": None,
                }
            ]

    def paste(self) -> None:
        if not self._clipboard:
            return
        offset = 5
        for ann in self._clipboard:
            bbox = ann["bbox"]
            new_bbox = [bbox[0] + offset, bbox[1] + offset, bbox[2] + offset, bbox[3] + offset]
            ann_copy = dict(ann)
            ann_copy["bbox"] = new_bbox
            self._add_annotation_from_dict(ann_copy)
        self._save_state()

    def save_labels(self) -> None:
        if not self.items:
            self.status.setText("没有可保存的标注。")
            return
        label_path = Path(image_to_label_path(self.image_path))
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        w, h = self.image_size
        if w <= 0 or h <= 0:
            self.status.setText("图片尺寸无效，无法保存。")
            return
        for item in self.items.values():
            x1, y1, x2, y2 = item.bbox()
            x1 = max(0.0, min(x1, w))
            y1 = max(0.0, min(y1, h))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{item.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.status.setText(f"已保存: {label_path}")
