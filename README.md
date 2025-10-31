# YOLOv8 ASL Alphabet Detection 🤟

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A computer vision project for detecting and recognizing American Sign Language (ASL) alphabet letters using YOLOv8/YOLO11 object detection.

![ASL Detection Demo](https://img.shields.io/badge/Status-Learning%20Project-yellow)

## Project Overview

This project trains a YOLOv8 model to detect hand signs representing ASL alphabet letters (A-Z) in images and real-time video streams. It's designed as a learning project to understand:

- Object detection using YOLOv8
- Dataset preparation and training
- Model evaluation and inference
- Real-time computer vision applications

## Dataset

**Dataset**: American Sign Language Letters v1
**Source**: [Roboflow Universe](https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters)
**License**: Public Domain
**Classes**: 26 (A-Z)
**Format**: YOLOv8 format with bounding box annotations

### Dataset Structure
```
American Sign Language Letters.v1-v1.yolov8/
├── data.yaml          # Dataset configuration
├── train/
│   ├── images/       # Training images
│   └── labels/       # Training annotations
├── valid/
│   ├── images/       # Validation images
│   └── labels/       # Validation annotations
└── test/
    ├── images/       # Test images
    └── labels/       # Test annotations
```

## Requirements

### Environment Setup

1. **Create Conda Environment** (Recommended):
```bash
conda create -n yolo11_ASL_Language python=3.11
conda activate yolo11_ASL_Language
```

2. **Install Dependencies**:
```bash
pip install ultralytics opencv-python
```

### System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- Webcam (for real-time detection)

## 📁 Project Structure

```
YOLO_ASL_Alphabet/
├── scripts/                    # Python scripts
│   ├── train_yolov8_asl.py    # Training script
│   ├── test_yolov8_asl.py     # Testing script
│   └── webcam_detection.py    # Real-time detection
├── examples/                   # Usage examples
│   └── quick_start.md         # Quick start guide
├── American Sign Language Letters.v1-v1.yolov8/  # Dataset
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/                       # Training outputs (gitignored)
├── outputs/                    # Inference outputs (gitignored)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Usage

### 1. Training the Model

Train a YOLOv8 model on the ASL dataset:

```bash
python scripts/train_yolov8_asl.py
```

**Training Features**:
- Uses YOLOv8 nano model (yolov8n.pt) for fast training
- 50 epochs with early stopping (patience=10)
- Batch size: 16
- Image size: 640x640
- Automatic Mixed Precision (AMP) training
- Data augmentation (HSV, translation, scale, flip, mosaic)
- Saves best and last model weights

**Output**:
- Model weights: `runs/asl_detection/yolov8_asl_training/weights/`
- Training metrics: `runs/asl_detection/yolov8_asl_training/`
- Validation results and visualizations

### 2. Testing the Model

#### Test on Full Dataset
```bash
python scripts/test_yolov8_asl.py --mode dataset --model runs/asl_detection/yolov8_asl_training/weights/best.pt
```

#### Test on Single Image
```bash
python scripts/test_yolov8_asl.py --mode image --source "American Sign Language Letters.v1-v1.yolov8/test/images/A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg" --conf 0.25
```

#### Test on Image Folder (Batch)
```bash
python scripts/test_yolov8_asl.py --mode batch --source path/to/images/ --conf 0.25
```

**Testing Options**:
- `--model`: Path to trained model weights (default: best.pt)
- `--mode`: Testing mode (image/dataset/batch)
- `--source`: Image path, data.yaml, or folder path
- `--conf`: Confidence threshold (default: 0.25)
- `--save`: Save output images with detections

### 3. Real-Time Webcam Detection

Run real-time ASL letter detection using your webcam:

```bash
python scripts/webcam_detection.py --model runs/asl_detection/yolov8_asl_training/weights/best.pt --conf 0.5
```

**Webcam Controls**:
- **`q` or `ESC`**: Quit the application
- **`s`**: Save current frame with detections
- **`c`**: Clear detected letters history
- **`+`**: Increase confidence threshold
- **`-`**: Decrease confidence threshold

**Options**:
- `--model`: Path to trained model (default: best.pt)
- `--conf`: Confidence threshold (default: 0.5)
- `--camera`: Camera device ID (default: 0)

**Features**:
- Real-time FPS display
- Letter history tracking (builds words from detected letters)
- Current detection with confidence score
- Adjustable confidence threshold on-the-fly
- Frame capture functionality

## Model Architecture

**YOLOv8 Nano (yolov8n.pt)**:
- Fastest and smallest YOLOv8 variant
- Suitable for real-time applications
- Good balance between speed and accuracy

**Other Available Models**:
- `yolov8s.pt` - Small (better accuracy, slower)
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (best accuracy, slowest)

To use a different model, edit the `model_size` variable in `scripts/train_yolov8_asl.py`.

## Training Configuration

Key hyperparameters in `scripts/train_yolov8_asl.py`:

```python
epochs = 50                  # Number of training epochs
batch = 16                   # Batch size
imgsz = 640                  # Input image size
lr0 = 0.01                   # Initial learning rate
patience = 10                # Early stopping patience
augmentation = {
    'hsv_h': 0.015,         # Hue augmentation
    'hsv_s': 0.7,           # Saturation augmentation
    'hsv_v': 0.4,           # Value augmentation
    'fliprl': 0.5,          # Horizontal flip probability
    'mosaic': 1.0,          # Mosaic augmentation
}
```

## Performance Metrics

The model is evaluated using:

- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of detected ground truth objects
- **Per-class metrics**: Individual performance for each letter

## Project Structure

```
YOLO_ASL Alphabet/
├── American Sign Language Letters.v1-v1.yolov8/  # Dataset
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── train_yolov8_asl.py          # Training script
├── test_yolov8_asl.py           # Testing script
├── webcam_detection.py          # Real-time detection
├── README.md                    # Documentation
├── runs/                        # Training outputs (created after training)
│   └── asl_detection/
│       └── yolov8_asl_training/
│           ├── weights/
│           │   ├── best.pt     # Best model
│           │   └── last.pt     # Last epoch
│           └── results.png     # Training curves
└── outputs/                     # Inference outputs (created during testing)
    ├── batch_results/          # Batch inference results
    └── webcam_captures/        # Saved webcam frames
```

## Learning Objectives

Through this project, you will learn:

1. **Dataset Preparation**: Understanding YOLO format annotations
2. **Model Training**: Training YOLOv8 with custom data
3. **Hyperparameter Tuning**: Adjusting learning rates, batch sizes, and augmentation
4. **Model Evaluation**: Interpreting mAP, precision, and recall metrics
5. **Inference**: Running predictions on images and video streams
6. **Real-Time CV**: Implementing webcam-based object detection
7. **Computer Vision Pipeline**: End-to-end ML workflow

## Tips for Better Performance

1. **Increase Training Epochs**: Try 100-150 epochs for better convergence
2. **Use Larger Models**: Switch to yolov8s or yolov8m for better accuracy
3. **Adjust Confidence Threshold**: Lower for more detections, higher for precision
4. **Data Augmentation**: Experiment with augmentation parameters
5. **Fine-tune on Your Data**: Collect your own ASL images for better personalization
6. **GPU Acceleration**: Use CUDA-enabled GPU for faster training

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: data.yaml not found`
**Solution**: Ensure the dataset folder path is correct in the training script

**Issue**: `CUDA out of memory`
**Solution**: Reduce batch size in training configuration or use CPU (slower)

**Issue**: Low FPS in webcam detection
**Solution**: Use yolov8n (nano) model and reduce image resolution

**Issue**: Poor detection accuracy
**Solution**: Train for more epochs, use a larger model, or adjust confidence threshold

## Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [ASL Dataset on Roboflow](https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters)
- [YOLO Object Detection Guide](https://blog.roboflow.com/yolov8-guide/)

## Credits

- **Dataset**: David Lee ([LinkedIn](https://www.linkedin.com/in/daviddaeshinlee/))
- **Framework**: Ultralytics YOLOv8
- **Project**: Educational learning project for computer vision

## License

- **Dataset**: Public Domain
- **Code**: Open source (MIT License compatible)

## Next Steps

1. **Train the model** with different configurations
2. **Experiment** with hyperparameters to improve accuracy
3. **Try larger models** (yolov8s, yolov8m) for better performance
4. **Collect your own data** to personalize the model
5. **Build a complete application** (e.g., ASL translator)
6. **Deploy the model** to mobile or edge devices

---

**Happy Learning!** 🚀

For questions or issues, refer to the YOLOv8 documentation or open an issue on GitHub.
#   C o m p u t e r - V i s i o n - A S L - L a n g u a g e  
 