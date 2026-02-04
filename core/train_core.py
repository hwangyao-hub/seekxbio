from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

from .utils import resolve_device, set_reproducibility


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _collect_images_from_dir(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return [str(p) for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def _collect_images_from_txt(txt_path: Path) -> list[str]:
    if not txt_path.exists():
        return []
    return [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_split(root: Path, entry: str) -> Path:
    p = Path(entry)
    return p if p.is_absolute() else (root / entry)


def _collect_images(root: Path, entry: str) -> list[str]:
    split_path = _resolve_split(root, entry)
    if split_path.suffix.lower() == ".txt":
        return _collect_images_from_txt(split_path)
    return _collect_images_from_dir(split_path)


def _names_list_from_data(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, list):
        return list(names)
    if isinstance(names, dict):
        max_id = max(int(k) for k in names.keys()) if names else -1
        return [names.get(i, names.get(str(i), "")) for i in range(max_id + 1)]
    return []


def _collect_used_class_ids(train_images: list[str]) -> list[int]:
    used: set[int] = set()
    for img in train_images:
        lbl = _image_to_label_path(Path(img))
        if lbl is None:
            continue
        for cid in _read_label_ids(lbl):
            used.add(cid)
    return sorted(used)


def _remap_label_file(src: Path, dst: Path, mapping: dict[int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return

    lines_out: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            old_id = int(float(parts[0]))
        except Exception:
            continue
        if old_id not in mapping:
            continue
        parts[0] = str(mapping[old_id])
        lines_out.append(" ".join(parts))
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            return
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _build_remap_dataset(
    root: Path,
    train_list: list[str],
    val_list: list[str],
    tmp_dir: Path,
    names_list: list[str],
    used_ids: list[int],
) -> tuple[str, str, list[str]]:
    mapping = {old: new for new, old in enumerate(used_ids)}
    new_names = [names_list[old] if old < len(names_list) else str(old) for old in used_ids]

    tmp_root = tmp_dir / "remap_dataset"
    tmp_root.mkdir(parents=True, exist_ok=True)

    def make_temp_image(img_path: Path) -> Path:
        try:
            rel = img_path.relative_to(root)
        except Exception:
            rel = Path("images") / img_path.name
        if "images" not in rel.parts:
            rel = Path("images") / rel
        tmp_img = tmp_root / rel
        _link_or_copy(img_path, tmp_img)
        return tmp_img

    temp_train: list[str] = []
    for img in train_list:
        img_path = Path(img)
        tmp_img = make_temp_image(img_path)
        src_lbl = _image_to_label_path(img_path)
        dst_lbl = _image_to_label_path(tmp_img)
        if src_lbl is not None and dst_lbl is not None:
            _remap_label_file(src_lbl, dst_lbl, mapping)
        temp_train.append(str(tmp_img))

    temp_val: list[str] = []
    for img in val_list:
        img_path = Path(img)
        tmp_img = make_temp_image(img_path)
        src_lbl = _image_to_label_path(img_path)
        dst_lbl = _image_to_label_path(tmp_img)
        if src_lbl is not None and dst_lbl is not None:
            _remap_label_file(src_lbl, dst_lbl, mapping)
        temp_val.append(str(tmp_img))

    train_txt = _write_list_file(temp_train, tmp_dir / "train_remap.txt")
    val_txt = _write_list_file(temp_val, tmp_dir / "val_remap.txt")
    return train_txt, val_txt, new_names


def _image_to_label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        return label_path
    # Fallback: same directory with .txt (best effort)
    return image_path.with_suffix(".txt")


def _read_label_ids(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    cls_ids: list[int] = []
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            cls_ids.append(int(float(parts[0])))
    except Exception:
        return []
    return cls_ids


def _trim_unused_tail_classes(data: dict, train_images: list[str]) -> dict:
    names = data.get("names")
    if not isinstance(names, (list, dict)):
        return data

    if isinstance(names, dict):
        # Convert to list by id order
        max_id = max(int(k) for k in names.keys()) if names else -1
        names_list = [names.get(i, names.get(str(i), "")) for i in range(max_id + 1)]
    else:
        names_list = list(names)

    max_used = -1
    for img in train_images:
        lbl = _image_to_label_path(Path(img))
        if lbl is None:
            continue
        for cid in _read_label_ids(lbl):
            if cid > max_used:
                max_used = cid

    if max_used < 0:
        return data

    # Only trim unused classes at the tail to avoid remapping ids.
    if max_used < len(names_list) - 1:
        names_list = names_list[: max_used + 1]
        data["names"] = names_list
    return data


def _write_list_file(items: list[str], out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(items) + "\n", encoding="utf-8")
    return str(out_path.resolve())


def _build_subset_yaml(
    data_yaml: str,
    limit_train: int,
    limit_val: int,
    seed: int,
    work_dir: Path,
    remap_classes: bool,
) -> str:
    try:
        import yaml
    except Exception as exc:
        raise ImportError(
            "PyYAML is required for subset YAML generation. "
            "Install with: pip install pyyaml"
        ) from exc

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", "")).expanduser()
    train_entry = data.get("train")
    val_entry = data.get("val")

    if not train_entry or not val_entry:
        raise ValueError("Dataset YAML must contain both 'train' and 'val' fields.")

    train_list = _collect_images(root, train_entry)
    val_list = _collect_images(root, val_entry)
    if not train_list:
        raise FileNotFoundError("No training images found with the given dataset YAML.")
    if not val_list:
        raise FileNotFoundError("No validation images found with the given dataset YAML.")

    rng = random.Random(seed)
    rng.shuffle(train_list)
    rng.shuffle(val_list)

    if limit_train > 0:
        train_list = train_list[: min(limit_train, len(train_list))]
    if limit_val > 0:
        val_list = val_list[: min(limit_val, len(val_list))]

    tmp_dir = work_dir / "tmp"

    subset_yaml = dict(data)
    if remap_classes:
        names_list = _names_list_from_data(subset_yaml)
        used_ids = _collect_used_class_ids(train_list)
        if used_ids:
            train_txt, val_txt, new_names = _build_remap_dataset(
                root=root,
                train_list=train_list,
                val_list=val_list,
                tmp_dir=tmp_dir,
                names_list=names_list,
                used_ids=used_ids,
            )
            subset_yaml["names"] = new_names
        else:
            train_txt = _write_list_file(train_list, tmp_dir / "train_subset.txt")
            val_txt = _write_list_file(val_list, tmp_dir / "val_subset.txt")
    else:
        train_txt = _write_list_file(train_list, tmp_dir / "train_subset.txt")
        val_txt = _write_list_file(val_list, tmp_dir / "val_subset.txt")
        subset_yaml = _trim_unused_tail_classes(subset_yaml, train_list)

    subset_yaml["path"] = ""  # use absolute paths in txt lists
    subset_yaml["train"] = train_txt
    subset_yaml["val"] = val_txt

    out_yaml = tmp_dir / "data_subset.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(subset_yaml, f, sort_keys=False)
    return str(out_yaml)


def _build_frozen_yaml(
    data_yaml: str,
    work_dir: Path,
    remap_classes: bool,
) -> str:
    try:
        import yaml
    except Exception as exc:
        raise ImportError(
            "PyYAML is required for dataset freezing. "
            "Install with: pip install pyyaml"
        ) from exc

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", "")).expanduser()
    train_entry = data.get("train")
    val_entry = data.get("val")

    if not train_entry or not val_entry:
        raise ValueError("Dataset YAML must contain both 'train' and 'val' fields.")

    train_list = _collect_images(root, train_entry)
    val_list = _collect_images(root, val_entry)
    if not train_list:
        raise FileNotFoundError("No training images found with the given dataset YAML.")
    if not val_list:
        raise FileNotFoundError("No validation images found with the given dataset YAML.")

    tmp_dir = work_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if remap_classes:
        names_list = _names_list_from_data(data)
        used_ids = _collect_used_class_ids(train_list)
        if used_ids:
            train_txt, val_txt, new_names = _build_remap_dataset(
                root=root,
                train_list=train_list,
                val_list=val_list,
                tmp_dir=tmp_dir,
                names_list=names_list,
                used_ids=used_ids,
            )
            data["names"] = new_names
        else:
            train_txt = _write_list_file(train_list, tmp_dir / "train_frozen.txt")
            val_txt = _write_list_file(val_list, tmp_dir / "val_frozen.txt")
    else:
        train_txt = _write_list_file(train_list, tmp_dir / "train_frozen.txt")
        val_txt = _write_list_file(val_list, tmp_dir / "val_frozen.txt")
        data = _trim_unused_tail_classes(data, train_list)

    data["path"] = ""
    data["train"] = train_txt
    data["val"] = val_txt

    out_yaml = tmp_dir / "data_frozen.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(out_yaml)


def train_yolov8(
    data_yaml: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 8,
    device: str = "auto",
    project: str = "runs/detect",
    name: str = "cells",
    seed: int = 42,
    resume: bool = False,
    limit_train_images: int = 0,
    limit_val_images: int = 0,
    remap_classes: bool = False,
    freeze_splits: bool = True,
) -> dict[str, str]:
    """
    Train a YOLOv8 model. Returns output paths (save_dir, best, last).
    """
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    set_reproducibility(seed)
    resolved_device = resolve_device(device)

    run_data_yaml = str(data_path)
    if limit_train_images > 0 or limit_val_images > 0:
        run_data_yaml = _build_subset_yaml(
            data_yaml=str(data_path),
            limit_train=limit_train_images,
            limit_val=limit_val_images,
            seed=seed,
            work_dir=Path(project).resolve() / name,
            remap_classes=remap_classes,
        )
    elif freeze_splits:
        run_data_yaml = _build_frozen_yaml(
            data_yaml=str(data_path),
            work_dir=Path(project).resolve() / name,
            remap_classes=remap_classes,
        )
    else:
        # Trim unused tail classes without altering label ids.
        try:
            import yaml
        except Exception as exc:
            raise ImportError(
                "PyYAML is required for automatic class trimming. "
                "Install with: pip install pyyaml"
            ) from exc
        with open(data_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        root = Path(data.get("path", "")).expanduser()
        train_entry = data.get("train")
        if train_entry:
            train_list = _collect_images(root, train_entry)
            if remap_classes:
                names_list = _names_list_from_data(data)
                used_ids = _collect_used_class_ids(train_list)
                if used_ids:
                    tmp_dir = (Path(project).resolve() / name / "tmp")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    train_txt, val_txt, new_names = _build_remap_dataset(
                        root=root,
                        train_list=train_list,
                        val_list=_collect_images(root, data.get("val")),
                        tmp_dir=tmp_dir,
                        names_list=names_list,
                        used_ids=used_ids,
                    )
                    data["names"] = new_names
                    data["path"] = ""
                    data["train"] = train_txt
                    data["val"] = val_txt
                    out_yaml = tmp_dir / "data_autoremap.yaml"
                    with open(out_yaml, "w", encoding="utf-8") as f:
                        yaml.safe_dump(data, f, sort_keys=False)
                    run_data_yaml = str(out_yaml)
            else:
                trimmed = _trim_unused_tail_classes(data, train_list)
                if trimmed.get("names") != data.get("names"):
                    tmp_dir = (Path(project).resolve() / name / "tmp")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    out_yaml = tmp_dir / "data_autotrim.yaml"
                    with open(out_yaml, "w", encoding="utf-8") as f:
                        yaml.safe_dump(trimmed, f, sort_keys=False)
                    run_data_yaml = str(out_yaml)

    model = YOLO(model_name)
    results = model.train(
        data=run_data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=resolved_device,
        project=project,
        name=name,
        resume=resume,
        seed=seed,
        deterministic=True,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"

    return {
        "save_dir": str(save_dir),
        "best": str(best) if best.exists() else "",
        "last": str(last) if last.exists() else "",
    }


def get_run_outputs(project: str, name: str) -> dict[str, str]:
    save_dir = Path(project) / name
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    return {
        "save_dir": str(save_dir),
        "best": str(best) if best.exists() else "",
        "last": str(last) if last.exists() else "",
    }


def train_yolov8_stream(
    data_yaml: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 8,
    device: str = "auto",
    project: str = "runs/detect",
    name: str = "cells",
    seed: int = 42,
    resume: bool = False,
    limit_train_images: int = 0,
    limit_val_images: int = 0,
    remap_classes: bool = False,
    freeze_splits: bool = True,
) -> "list[str] | None":
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--data",
        data_yaml,
        "--model",
        model_name,
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--device",
        device,
        "--project",
        project,
        "--name",
        name,
        "--seed",
        str(seed),
    ]
    if resume:
        cmd.append("--resume")
    if limit_train_images > 0:
        cmd += ["--limit-train", str(limit_train_images)]
    if limit_val_images > 0:
        cmd += ["--limit-val", str(limit_val_images)]
    if remap_classes:
        cmd += ["--remap-classes"]
    if not freeze_splits:
        cmd += ["--no-freeze-splits"]

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if proc.stdout is None:
        return

    for line in proc.stdout:
        yield line

    proc.wait()
    if proc.returncode != 0:
        yield f"\n[ERROR] Training process exited with code {proc.returncode}\n"
