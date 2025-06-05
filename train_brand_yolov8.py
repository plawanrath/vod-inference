# scripts/train_brand_yolov8.py
#!/usr/bin/env python
"""
Fine-tune a YOLOv8 detector on a custom brand dataset.
Includes an option to download datasets from Kaggle.

Example for training:
    python train_brand_yolov8.py --data brands.yaml --model yolov8m.pt \
        --epochs 50 --batch 16 --imgsz 640 --device mps

Example for downloading a Kaggle dataset:
    python train_brand_yolov8.py \
        --download_kaggle_dataset_slug "momotabanerjee/brand-logo-recognition-dataset" \
        --download_output_dir "./downloaded_datasets"

The training dataset should follow the standard YOLO directory layout::

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
import os
# Note: 'zipfile' might be needed if Kaggle API doesn't handle unzip for all cases,
# but dataset_download_files with unzip=True usually handles it.
# import zipfile 
from ultralytics import YOLO

def _download_kaggle_dataset(dataset_slug: str, output_dir: str) -> None:
    """
    Downloads and unzips a dataset from Kaggle.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: The 'kaggle' library is not installed. Please install it with 'pip install kaggle'.")
        print("Also ensure your Kaggle API credentials (kaggle.json or environment variables) are set up.")
        return

    print(f"Attempting to download Kaggle dataset: {dataset_slug} to {output_dir}")

    try:
        api = KaggleApi()
        api.authenticate() # Checks for credentials
    except Exception as e:
        print(f"Error: Kaggle API authentication failed. Ensure kaggle.json is in ~/.kaggle/ or KAGGLE_USERNAME/KAGGLE_KEY env vars are set.")
        print(f"Details: {e}")
        return

    # Specific path for this dataset download
    dataset_specific_download_path = os.path.join(output_dir, dataset_slug.split('/')[-1])
    os.makedirs(dataset_specific_download_path, exist_ok=True)

    try:
        print(f"Downloading files to {dataset_specific_download_path}...")
        api.dataset_download_files(dataset_slug, path=dataset_specific_download_path, unzip=True)
        print(f"Dataset {dataset_slug} downloaded and unzipped to {dataset_specific_download_path}")
        
        print("\nNext steps for using this dataset:")
        print(f"1. Inspect the downloaded files in '{dataset_specific_download_path}'.")
        print(f"2. Reorganize the images and labels into the required YOLO format (train/val splits with images/labels subfolders).")
        print(f"3. Create or update a .yaml configuration file (like brands.yaml) to point to the root of this prepared dataset and list its class names accurately.")

    except Exception as e:
        print(f"Error downloading or unzipping dataset {dataset_slug}: {e}")
        print("Please ensure the dataset slug is correct, you have permissions, and enough disk space.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on brand images or download Kaggle datasets.")
    
    # Training arguments
    p.add_argument("--data", help="Dataset YAML config for training (e.g., brands.yaml)")
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
        help="Pretrained checkpoint to start training from",
    )
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--batch", type=int, default=16, help="Batch size for training")
    p.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    p.add_argument(
        "--device", default="auto", help="Computation device for training ('cpu', 'mps', 'cuda', etc.)"
    )
    p.add_argument("--name", default="yolov8_brands", help="Run name for logging training results")

    # Kaggle Download arguments
    p.add_argument(
        "--download_kaggle_dataset_slug",
        type=str,
        default=None,
        help="Kaggle dataset slug (e.g., 'username/dataset-name') to download. "
             "Example: 'momotabanerjee/brand-logo-recognition-dataset'. "
             "If provided, the script will only download and then exit."
    )
    p.add_argument(
        "--download_output_dir",
        type=str,
        default="./downloaded_datasets",
        help="Directory where Kaggle datasets will be downloaded and unzipped."
    )
    return p.parse_args()


def run_training(args: argparse.Namespace) -> None:
    if not args.data:
        print("Error: --data (dataset YAML config) is required for training.")
        return
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
    args = _parse_args()

    if args.download_kaggle_dataset_slug:
        _download_kaggle_dataset(
            args.download_kaggle_dataset_slug,
            args.download_output_dir
        )
        print(f"\nDownload process for '{args.download_kaggle_dataset_slug}' initiated.")
        print("Please check console messages for status and next steps after completion.")
        return  # Exit after attempting download

    # If not downloading, proceed with training
    print("Proceeding with training...")
    run_training(args)


if __name__ == "__main__":
    main()