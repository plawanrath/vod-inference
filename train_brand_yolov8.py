#!/usr/bin/env python
"""
Fine-tune a YOLOv8 detector on a custom brand dataset
(converted OpenLogo in ./openlogo_yolo by default).

Training:
    python train_brand_yolov8.py \
        --epochs 50 --batch 16 --imgsz 640 --device mps \
        --name openlogo_yolov8m

    python train_brand_yolov8.py --model yolov8m.pt --device mps --epochs 50
    
Custom dataset:
    python train_brand_yolov8.py --data path/to/your.yaml ...

Download only (Kaggle):
    python train_brand_yolov8.py \
       --download_kaggle_dataset_slug "momotabanerjee/brand-logo-recognition-dataset"
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Optional Kaggle download helper
# ─────────────────────────────────────────────────────────────────────────────
def _download_kaggle_dataset(dataset_slug: str, output_dir: str) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("❌  pip install kaggle  and place kaggle.json in ~/.kaggle/")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        sys.exit(f"❌  Kaggle API auth failed: {e}")

    out_path = Path(output_dir) / dataset_slug.split("/")[-1]
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading {dataset_slug} to {out_path} ...")
    api.dataset_download_files(dataset_slug, path=str(out_path), unzip=True)
    print("✅  Done. Please reorganise into YOLO format before training.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", help="Dataset YAML (defaults to ./openlogo_yolo/brands.yaml)")
    p.add_argument("--data_root", default="openlogo_yolo", help="Root folder if --data not given")
    p.add_argument("--model", default="yolov8m.pt",
                   choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                            "yolov8l.pt", "yolov8x.pt"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--device", default="auto", help="'cpu', 'mps', 'cuda', etc.")
    p.add_argument("--name",   default="yolov8_brands")
    p.add_argument("--resume", action="store_true",
                   help="Resume from runs/detect/<name>/weights/last.pt if exists")

    # Kaggle
    p.add_argument("--download_kaggle_dataset_slug")
    p.add_argument("--download_output_dir", default="./downloaded_datasets")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Sanity-check helpers
# ─────────────────────────────────────────────────────────────────────────────
def _auto_dataset_yaml(data_arg: str | None, root: str) -> Path:
    if data_arg:
        return Path(data_arg).expanduser()
    default_yaml = Path(root) / "brands.yaml"
    if not default_yaml.exists():
        sys.exit("❌  Dataset YAML not found. Pass --data <yaml> or place brands.yaml "
                 "inside the specified --data_root.")
    return default_yaml


def _check_yolo_dirs(yaml_path: Path):
    import yaml
    cfg = yaml.safe_load(yaml_path.read_text())
    for split in ("train", "val"):
        img_dir = yaml_path.parent / cfg["path"] / cfg[split]
        if not img_dir.exists():
            sys.exit(f"❌  {split} images dir missing: {img_dir}")
    print("✅  Dataset folders found, proceeding to training…")


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def _train(opt: argparse.Namespace):
    data_yaml = _auto_dataset_yaml(opt.data, opt.data_root)
    _check_yolo_dirs(data_yaml)

    model = YOLO(opt.model)
    resume_path = Path(f"runs/detect/{opt.name}/weights/last.pt")
    resume = opt.resume and resume_path.exists()

    print(f"📚  Training {opt.model} on {data_yaml} "
          f"{'(resume)' if resume else ''} for {opt.epochs} epochs…")

    model.train(
        data=str(data_yaml),
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        name=opt.name,
        resume=resume
    )
    print("🏁  Training finished.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    opt = _args()

    if opt.download_kaggle_dataset_slug:
        _download_kaggle_dataset(opt.download_kaggle_dataset_slug, opt.download_output_dir)
        return

    _train(opt)


if __name__ == "__main__":
    main()
