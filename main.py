import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication


def _project_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _launch_xanylabeling(argv: list[str]) -> None:
    # Run X-AnyLabeling inside current process (used in packaged mode).
    xany_root = _project_root() / "third_party" / "X-AnyLabeling"
    sys.path.insert(0, str(xany_root))
    from anylabeling import app as xany_app  # type: ignore
    sys.argv = argv
    xany_app.main()


def main():
    if "--xanylabeling" in sys.argv:
        idx = sys.argv.index("--xanylabeling")
        _launch_xanylabeling([sys.argv[0]] + sys.argv[idx + 1 :])
        return
    app = QApplication(sys.argv)
    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
