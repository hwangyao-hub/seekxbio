from __future__ import annotations

from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def _list_labels(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*.txt")]


def _stem_map(paths: list[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


def _read_label_file(path: Path) -> list[int]:
    cls_ids: list[int] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            cls_ids.append(int(float(parts[0])))
    except Exception:
        # If a single label file is malformed, skip it but continue scanning.
        return []
    return cls_ids


def _detect_layout(root: Path) -> tuple[str, list[str]]:
    # Layout A: images/ & labels/ at root, optionally with split subfolders.
    if (root / "images").exists() and (root / "labels").exists():
        if any((root / "images" / s).exists() for s in ("train", "val", "test")):
            return "images_labels_split", ["train", "val", "test"]
        return "images_labels_flat", ["all"]

    # Layout B: split folders contain images/ & labels/ (e.g., train/images).
    if any((root / s / "images").exists() for s in ("train", "val", "test")):
        return "split_images_labels", ["train", "val", "test"]

    return "unknown", ["train", "val", "test"]


def scan_dataset(dataset_root: str) -> dict[str, object]:
    """
    Scan a YOLO-style dataset and report stats and mismatches.
    Supported structures:
      - images/{train,val,test}/ + labels/{train,val,test}/
      - images/ + labels/ (flat, no split)
      - train/images + train/labels (and optionally val/test)
    """
    root = Path(dataset_root)
    layout, splits = _detect_layout(root)
    stats = {
        "dataset_root": str(root),
        "layout": layout,
        "split_stats": {},
        "total_images": 0,
        "total_labels": 0,
        "missing_labels": 0,
        "missing_images": 0,
        "class_counts": {},
    }

    global_counter: Counter[int] = Counter()

    for split in splits:
        if layout == "images_labels_split":
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
        elif layout == "images_labels_flat":
            img_dir = root / "images"
            lbl_dir = root / "labels"
        elif layout == "split_images_labels":
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"
        else:
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split

        imgs = _list_images(img_dir)
        lbls = _list_labels(lbl_dir)

        img_map = _stem_map(imgs)
        lbl_map = _stem_map(lbls)

        missing_lbl = sorted(set(img_map.keys()) - set(lbl_map.keys()))
        missing_img = sorted(set(lbl_map.keys()) - set(img_map.keys()))

        # Count class ids in this split
        split_counter: Counter[int] = Counter()
        for lbl_path in lbls:
            split_counter.update(_read_label_file(lbl_path))
        global_counter.update(split_counter)

        stats["split_stats"][split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "missing_labels": len(missing_lbl),
            "missing_images": len(missing_img),
        }

        stats["total_images"] += len(imgs)
        stats["total_labels"] += len(lbls)
        stats["missing_labels"] += len(missing_lbl)
        stats["missing_images"] += len(missing_img)

    stats["class_counts"] = {str(k): v for k, v in sorted(global_counter.items())}
    return stats


def scan_dataset_from_yaml(yaml_path: str) -> dict[str, object]:
    try:
        import yaml
    except Exception as exc:
        raise ImportError(
            "PyYAML is required to read dataset YAML. Install with: pip install pyyaml"
        ) from exc

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data.get("path", "")).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return scan_dataset(str(root))


def build_val_split(dataset_root: str, ratio: float = 0.1, seed: int = 42) -> dict[str, int]:
    """
    Build val split from train/images + train/labels in-place (standard layout).
    Returns counts: train_images, val_images.
    """
    root = Path(dataset_root)
    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"
    val_img = root / "val" / "images"
    val_lbl = root / "val" / "labels"

    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    images = [p for p in train_img.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    pairs = []
    for img in images:
        lbl = train_lbl / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))

    if not pairs:
        return {"train_images": 0, "val_images": 0}

    import random
    random.seed(seed)
    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * ratio))
    val_pairs = pairs[:val_count]

    # Clear existing val
    for p in val_img.rglob("*"):
        if p.is_file():
            p.unlink()
    for p in val_lbl.rglob("*"):
        if p.is_file():
            p.unlink()

    for img, lbl in val_pairs:
        (val_img / img.name).write_bytes(img.read_bytes())
        (val_lbl / lbl.name).write_bytes(lbl.read_bytes())

    return {"train_images": len(pairs), "val_images": len(val_pairs)}
