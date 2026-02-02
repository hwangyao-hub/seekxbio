# Copilot / AI Agent Instructions — Microscopy YOLOv8

Purpose: help an AI code assistant be productive quickly in this repo by describing architecture, workflows, conventions, and key files.

- **Big picture**: This repo offers two UI entry points and a core processing layer:
  - `app.py` — Gradio web UI for dataset scanning, streamed training, and single-image inference.
  - `main.py` ➜ `ui/` — PySide6 desktop UI (`ui/main_window.py`) alternative.
  - `core/` — Core domain logic: `train_core.py` (training orchestration & remapping), `infer_core.py` (single-image inference, rendering, exports), `dataset_utils.py` (dataset scanning/helpers).
  - `scripts/` — CLI wrappers that call `core/*`: `scripts/train.py` (used by `train_yolov8_stream`) and `scripts/infer_and_count.py`.

- **Primary flows & where to edit**:
  - Training: high-level logic is in `core/train_core.py` (`train_yolov8`, `train_yolov8_stream`). CLI args are implemented in `scripts/train.py`.
    - `train_yolov8_stream()` launches `scripts/train.py` via `subprocess` and yields stdout lines — modify that if changing streaming behavior.
    - Dataset trimming/remapping behavior (auto-trim vs `--remap-classes`) lives in `_build_subset_yaml` and helpers — this repo uses file-list txts and temporary `runs/detect/<name>/tmp` for subsets.
  - Inference: `core/infer_core.py` runs `ultralytics.YOLO.predict`, converts results into detections, renders overlays, and returns `(PIL.Image, counts, total)`; `scripts/infer_and_count.py` saves outputs to `outputs/infer`.

- **Data & file conventions** (important)
  - Dataset YAML (`data/cell.yaml`) must contain `path`, `train`, `val`, and `names`. `path` is the dataset root and `train`/`val` may be directories or `.txt` lists.
  - Image/label pairing: preferred layout is `.../images/...` and `.../labels/...` with YOLO `.txt` label files. If `images` is present, label path is derived by replacing `images` → `labels` and `.jpg`→`.txt`.
  - Temporary remapped datasets are created under `runs/detect/<name>/tmp` and use absolute paths via generated `.txt` lists.

- **Common commands / developer workflows** (copyable)
  - Install deps (CPU example):
    ```bash
    pip install -r requirements.txt
    pip install ultralytics torch torchvision pillow gradio pyyaml
    ```
  - Train (CLI):
    ```bash
    python scripts/train.py --data data/cell.yaml --model yolov8n.pt --epochs 100 --imgsz 640
    ```
  - Streamed training (from UI code): `core.train_core.train_yolov8_stream(...)` launches the above CLI; modify `scripts/train.py` only when changing CLI flags.
  - Inference + counting CLI:
    ```bash
    python scripts/infer_and_count.py --weights runs/detect/cells/weights/best.pt --source path/to/image.png
    ```
  - Launch Gradio UI: `python app.py`  — it calls `core.*` functions directly for dataset scan, inference, and streaming train.
  - Launch Qt UI: `python main.py` (requires `PySide6`).

- **Patterns & gotchas for edits**
  - Do not assume labels are contiguous; the code supports trimming/remapping. If changing label-handling, update both `_build_remap_dataset` and `_remap_label_file`.
  - `train_yolov8_stream` relies on `sys.executable` and repository `cwd` when launching `scripts/train.py`. Keep CLI args stable or update both caller and script.
  - Inference rendering uses PIL and attempts several local fonts; tests on Windows should verify font fallback paths.
  - Exports: `export_xanylabeling_json` produces AnyLabeling-compatible JSON files under `outputs/xanylabeling` — reuse this when adding integrations.

- **External dependencies & integration points**
  - Core ML: `ultralytics` (YOLOv8) and `torch` / `torchvision`.
  - UI: `gradio` (for `app.py`) and `PySide6` (for desktop `main.py`).
  - Optional: `pyyaml` is used for dataset YAML manipulation (auto-trim/remap). If missing, functions raise informative ImportError.

- **Where to start for common tasks**
  - Change training hyperparams: edit CLI defaults (`scripts/train.py`) or modify calls in `core/train_core.py`/`app.py` UI bindings.
  - Improve inference visualization or add outputs: edit `core/infer_core.py` (`render_detections`, `render_counts_overlay`, `export_xanylabeling_json`).
  - Add dataset validation rules: extend `core/dataset_utils.py` and its `scan_dataset` helper used by `app.py`.

If anything here is unclear or you want this shortened/expanded for a particular agent role (refactorer, test-writer, doc-updater), tell me which sections to adjust.
