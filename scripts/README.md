# Scripts Directory

This folder contains all the Python scripts for training, testing, and inference.

## Files

- **`train_yolov8_asl.py`** - Training script for YOLOv8/YOLO11 model
- **`test_yolov8_asl.py`** - Testing and evaluation script
- **`webcam_detection.py`** - Real-time webcam detection script

## Usage

### Training
```bash
cd ..
python scripts/train_yolov8_asl.py
```

### Testing
```bash
python scripts/test_yolov8_asl.py --mode dataset
```

### Webcam Detection
```bash
python scripts/webcam_detection.py --conf 0.5
```
