from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QSettings


def settings() -> QSettings:
    return QSettings("MicroscopyAI", "MicroscopyAI")


def get_setting(key: str, default: str = "") -> str:
    return str(settings().value(key, default))


def set_setting(key: str, value: str) -> None:
    settings().setValue(key, value)


def find_latest_model(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = list(base_dir.rglob("best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def project_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def default_yaml_path() -> Path:
    return project_root() / "data" / "cell.yaml"


def default_weights_path() -> Path | None:
    return find_latest_model(project_root() / "runs" / "detect")


def read_version() -> str:
    version_path = project_root() / "VERSION"
    if not version_path.exists():
        return "0.0.0"
    return version_path.read_text(encoding="utf-8").strip() or "0.0.0"
