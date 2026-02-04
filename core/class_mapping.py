from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .constants import CLASS_NAMES_CN


def _mapping_path(project_root: Path) -> Path:
    return project_root / "outputs" / "config" / "class_mapping.json"


def _default_english_names(project_root: Path) -> dict[int, str]:
    yaml_path = project_root / "data" / "cell.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def default_mapping_rows(project_root: Path) -> list[dict[str, object]]:
    english = _default_english_names(project_root)
    ids = sorted(set(english.keys()) | set(CLASS_NAMES_CN.keys()))
    rows: list[dict[str, object]] = []
    for cid in ids:
        rows.append(
            {
                "id": cid,
                "en": english.get(cid, ""),
                "cn": CLASS_NAMES_CN.get(cid, ""),
            }
        )
    return rows


def load_class_mapping_rows(project_root: Path) -> list[dict[str, object]]:
    path = _mapping_path(project_root)
    if not path.exists():
        return default_mapping_rows(project_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_mapping_rows(project_root)
    rows = data.get("classes")
    if not isinstance(rows, list):
        return default_mapping_rows(project_root)
    cleaned: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            cid = int(row.get("id"))
        except Exception:
            continue
        cleaned.append(
            {
                "id": cid,
                "en": str(row.get("en", "")).strip(),
                "cn": str(row.get("cn", "")).strip(),
            }
        )
    if not cleaned:
        return default_mapping_rows(project_root)
    return sorted(cleaned, key=lambda r: int(r["id"]))


def save_class_mapping_rows(
    project_root: Path,
    rows: Iterable[dict[str, object]],
) -> str:
    path = _mapping_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized: list[dict[str, object]] = []
    for row in rows:
        try:
            cid = int(row.get("id"))
        except Exception:
            continue
        normalized.append(
            {
                "id": cid,
                "en": str(row.get("en", "")).strip(),
                "cn": str(row.get("cn", "")).strip(),
            }
        )
    normalized = sorted(normalized, key=lambda r: int(r["id"]))
    payload = {"version": 1, "classes": normalized}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def rows_to_maps(rows: Iterable[dict[str, object]]) -> tuple[dict[int, str], dict[int, str]]:
    en_map: dict[int, str] = {}
    cn_map: dict[int, str] = {}
    for row in rows:
        try:
            cid = int(row.get("id"))
        except Exception:
            continue
        en = str(row.get("en", "")).strip()
        cn = str(row.get("cn", "")).strip()
        if en:
            en_map[cid] = en
        if cn:
            cn_map[cid] = cn
    return en_map, cn_map
