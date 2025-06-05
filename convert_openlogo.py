#!/usr/bin/env python
"""
Convert QMUL-OpenLogo (Pascal-VOC XML) to YOLOv8 format.

• Source dirs: Annotations/, JPEGImages/, ImageSets/
• Destination:  openlogo_yolo/{train,val}/{images,labels}
• Generates:    openlogo_yolo/brands.yaml

python convert_openlogo.py
"""

import os, shutil, xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC_IMG = ROOT / "openlogo" / "JPEGImages"
SRC_XML = ROOT / "openlogo" / "Annotations"

# -------- locate list files -------------------------------------------------
MAIN_DIR = ROOT / "openlogo" / "ImageSets" / "Main"

LIST_TRAIN = MAIN_DIR / "train_test" / "train_all.txt"   # 9 k images
LIST_VAL   = MAIN_DIR / "train_test" / "test_all.txt"                   # 5 k images
assert LIST_TRAIN.exists(), f"Train list not found: {LIST_TRAIN}"
assert LIST_VAL.exists(),   f"Val   list not found: {LIST_VAL}"

DEST = ROOT / "openlogo_yolo"
(DEST / "train" / "images").mkdir(parents=True, exist_ok=True)
(DEST / "train" / "labels").mkdir(parents=True, exist_ok=True)
(DEST / "val"   / "images").mkdir(parents=True, exist_ok=True)
(DEST / "val"   / "labels").mkdir(parents=True, exist_ok=True)

# -------- class-id map ------------------------------------------------------
classes = []           # keep insertion order
def cid(name: str) -> int:
    if name not in classes:
        classes.append(name)
    return classes.index(name)

def voc2yolo(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    w = int(root.findtext("size/width"));  h = int(root.findtext("size/height"))
    lines = []
    for obj in root.iter("object"):
        name = obj.findtext("name")
        bb   = obj.find("bndbox")
        xmin, ymin, xmax, ymax = map(float, (bb.findtext("xmin"), bb.findtext("ymin"),
                                             bb.findtext("xmax"), bb.findtext("ymax")))
        x_c, y_c = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw,  bh  = (xmax - xmin) / w,      (ymax - ymin) / h
        lines.append(f"{cid(name)} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
    return lines

def convert(list_file: Path, split: str):
    img_out = DEST / split / "images"
    lbl_out = DEST / split / "labels"
    ids = [x.strip() for x in list_file.open()]
    for img_id in tqdm(ids, desc=f"{split}"):
        xml = SRC_XML / f"{img_id}.xml"
        img = SRC_IMG / f"{img_id}.jpg"
        if not xml.exists() or not img.exists():
            continue
        yolo = voc2yolo(xml)
        if not yolo:
            continue            # skip images with no objects
        (lbl_out / f"{img_id}.txt").write_text("\n".join(yolo))
        shutil.copy2(img, img_out / f"{img_id}.jpg")

convert(LIST_TRAIN, "train")
convert(LIST_VAL,   "val")

# -------- write dataset yaml -----------------------------------------------
yaml_path = DEST / "brands.yaml"
yaml_path.write_text(
    f"path: {DEST}\ntrain: train/images\nval: val/images\nnames:\n" +
    "".join(f"  {i}: {n}\n" for i, n in enumerate(classes))
)
print(f"✅ Done. {len(classes)} classes → {yaml_path}")
