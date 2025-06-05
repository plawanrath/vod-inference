# scripts/train_brand_yolov8.py
#!/usr/bin/env python
"""
Fine-tune a YOLOv8 detector on a custom brand dataset.

Example
    python train_brand_yolov8.py --data brands.yaml --model yolov8m.pt \
        --epochs 50 --batch 16 --imgsz 640 --device mps

The dataset should follow the standard YOLO directory layout::

    dataset/
        train/
            images/
            labels/
        val/
            images/
            labels/

Each label file uses the format ``<class_id> <x_center> <y_center> <width> <height>``
with values normalized between 0 and 1. A ``brands.yaml`` file lists the class
names and paths, e.g.::

    path: ./dataset
    train: train/images
    val: val/images
    names:
      0: CocaCola
      1: Pepsi
      2: Nike
      3: Apple

Adjust the ``names`` section to match your brand classes.
"""
import argparse
from ultralytics import YOLO


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on brand images")
    p.add_argument("--data", required=True, help="Dataset YAML config")
    p.add_argument(
        "--model",
        default="yolov8m.pt",
        choices=[
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        ],
        help="Pretrained checkpoint to start from",
    )
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    p.add_argument(
        "--device", default="auto", help="Computation device ('cpu', 'mps', 'cuda', etc.)"
    )
    p.add_argument("--name", default="yolov8_brands", help="Run name for logging")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
    )


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
