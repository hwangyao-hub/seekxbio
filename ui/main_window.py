from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QStackedWidget,
)

from ui.top_bar import TopBar
from ui.side_bar import SideBar
from ui.pages.dashboard_page import DashboardPage
from ui.pages.inference_page import InferencePage
from ui.pages.training_page import TrainingPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscopy AI System")
        self.resize(1200, 800)

        # ===== Central Layout =====
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # ===== Top Bar =====
        self.top_bar = TopBar()
        root_layout.addWidget(self.top_bar)

        # ===== Main Area =====
        main_area = QWidget()
        main_layout = QHBoxLayout(main_area)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.sidebar = SideBar()
        self.stack = QStackedWidget()

        # Pages
        self.dashboard_page = DashboardPage()
        self.inference_page = InferencePage()
        self.training_page = TrainingPage()

        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.inference_page)
        self.stack.addWidget(self.training_page)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stack, 1)

        root_layout.addWidget(main_area, 1)

        # ===== Signals =====
        self.sidebar.page_changed.connect(self.stack.setCurrentIndex)
        self.inference_page.model_changed.connect(self.top_bar.set_model)
        self.inference_page.device_changed.connect(self.top_bar.set_device)
        self.dashboard_page.dataset_root_changed.connect(self.training_page.set_dataset_root)
        self.dashboard_page.dataset_root_changed.connect(self.inference_page.set_dataset_root)

        # Initialize top bar with persisted values
        self.top_bar.set_model(self.inference_page.weights_edit.text().strip() or "-")
        self.top_bar.set_device(self.inference_page.device.currentText())
