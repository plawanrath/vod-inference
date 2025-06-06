Dataset used to fine-tune YOLOv*: https://qmul-openlogo.github.io 

# Steps to setup the dataset for fine tuning:

- Download the dataset.
- If using openlogo then download https://qmul-openlogo.github.io into openlogo folder
- Run the `convert_openlogo.py` to convert the dataset into YLO format. This will store YOLO format dataset in openlogo_yolo folder

# Fine-tuning steps

- Ensure that you have dataset available in YOLO format.
- Run the following:
```
python train_brand_yolov8.py --model yolov8m.pt --device mps --epochs 50 
```