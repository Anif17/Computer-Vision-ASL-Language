# YOLOv8 ASL Alphabet Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A computer vision project for detecting and recognizing American Sign Language (ASL) alphabet letters using YOLOv8/YOLO11 object detection.

![Status](https://img.shields.io/badge/Status-Learning%20Project-yellow)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
  - [Real-Time Detection](#real-time-detection)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project trains a YOLOv8/YOLO11 model to detect hand signs representing ASL alphabet letters (A-Z) in images and real-time video streams.

**Learning Goals:**
- Object detection using YOLOv8/YOLO11
- Dataset preparation and training workflows
- Model evaluation and inference techniques
- Real-time computer vision applications

---

## Features

- **Training Pipeline**: Complete training script with customizable hyperparameters
- **Testing Suite**: Evaluate on single images, batches, or full test set
- **Real-Time Detection**: Webcam-based ASL letter recognition
- **Letter Tracking**: Build words from detected letters in real-time
- **GPU Acceleration**: CUDA support for faster training and inference
- **Comprehensive Documentation**: Setup guides, examples, and tutorials

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | American Sign Language Letters v1 |
| **Source** | [Roboflow Universe](https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters) |
| **License** | Public Domain |
| **Classes** | 26 (A-Z) |
| **Format** | YOLOv8 with bounding box annotations |
| **Splits** | Train / Validation / Test |

### Dataset Structure

```
Dataset ASL Languauge/
├── data.yaml          # Dataset configuration
├── train/
│   ├── images/       # Training images
│   └── labels/       # Training annotations (YOLO format)
├── valid/
│   ├── images/       # Validation images
│   └── labels/       # Validation annotations
└── test/
    ├── images/       # Test images
    └── labels/       # Test annotations
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Webcam (for real-time detection)

### Quick Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/YOLO_ASL_Alphabet.git
cd YOLO_ASL_Alphabet
```

2. **Create environment**

```bash
conda create -n yolo_asl python=3.11
conda activate yolo_asl
```

3. **Install dependencies**

```bash
# For GPU (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# For CPU only
pip install -r requirements.txt
```

4. **Download dataset**

Visit [Roboflow Universe](https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters) and download the dataset in YOLOv8 format. Extract to project root.

For detailed setup instructions, see [SETUP.md](SETUP.md).

---

## Usage

### Training

Train a YOLOv8 model on the ASL dataset:

```bash
python scripts/train_yolov8_asl.py
```

**Training Features:**
- YOLOv8 nano model (fast training)
- 50 epochs with early stopping
- Batch size: 16
- Image size: 640x640
- Data augmentation enabled
- Automatic Mixed Precision (AMP)

**Output:**
- Model weights: `runs/asl_detection/yolov8_asl_training/weights/best.pt`
- Training metrics and curves

### Testing

#### Test on Full Dataset

```bash
python scripts/test_yolov8_asl.py --mode dataset
```

#### Test on Single Image

```bash
python scripts/test_yolov8_asl.py \
  --mode image \
  --source "Dataset ASL Languauge/test/images/A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg" \
  --conf 0.25
```

#### Batch Testing

```bash
python scripts/test_yolov8_asl.py \
  --mode batch \
  --source path/to/images/ \
  --conf 0.25
```

**Options:**
- `--model`: Path to trained weights
- `--mode`: Testing mode (image/dataset/batch)
- `--source`: Image path, data.yaml, or folder
- `--conf`: Confidence threshold (0.0-1.0)
- `--save`: Save annotated outputs

### Real-Time Detection

Run real-time ASL detection using your webcam:

```bash
python scripts/webcam_detection.py --conf 0.5
```

**Keyboard Controls:**
- `q` or `ESC` - Quit application
- `s` - Save current frame
- `c` - Clear letter history
- `+` - Increase confidence threshold
- `-` - Decrease confidence threshold

**Features:**
- Real-time FPS display
- Letter history tracking
- Confidence scores
- Adjustable thresholds

---

## Project Structure

```
YOLO_ASL_Alphabet/
├── scripts/                    # Python scripts
│   ├── train_yolov8_asl.py    # Training script
│   ├── test_yolov8_asl.py     # Testing script
│   └── webcam_detection.py    # Real-time detection
│
├── examples/                   # Usage examples
│   └── quick_start.md         # Quick start guide
│
├── Dataset ASL Languauge/      # Dataset (not in repo)
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
│
├── runs/                       # Training outputs (gitignored)
│   └── asl_detection/
│       └── yolov8_asl_training/
│           ├── weights/
│           │   ├── best.pt    # Best model
│           │   └── last.pt    # Last epoch
│           └── results.png    # Training curves
│
├── outputs/                    # Inference outputs (gitignored)
│   ├── batch_results/
│   └── webcam_captures/
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
├── README.md                  # This file
└── SETUP.md                   # Detailed setup guide
```

---

## Model Architecture

### Available Models

| Model | Size | Parameters | Speed | Accuracy |
|-------|------|------------|-------|----------|
| yolo11n.pt | ~6 MB | 2.6M | Fastest | Good |
| yolo11s.pt | ~19 MB | 9.4M | Fast | Better |
| yolo11m.pt | ~40 MB | 20.1M | Medium | Great |
| yolo11l.pt | ~50 MB | 25.3M | Slow | Excellent |
| yolo11x.pt | ~110 MB | 56.9M | Slowest | Best |

**Default:** `yolo11n.pt` (nano) for faster training

To use a different model, edit `model_size` in `scripts/train_yolov8_asl.py`:

```python
model_size = "yolo11s.pt"  # Change to desired model
```

### Training Configuration

Key hyperparameters in `scripts/train_yolov8_asl.py`:

```python
epochs = 50                  # Training epochs
batch = 16                   # Batch size
imgsz = 640                  # Input image size
lr0 = 0.01                   # Initial learning rate
patience = 10                # Early stopping patience
workers = 2                  # Data loader workers

# Data augmentation
hsv_h = 0.015               # Hue augmentation
hsv_s = 0.7                 # Saturation augmentation
fliplr = 0.5                # Horizontal flip probability
mosaic = 1.0                # Mosaic augmentation
```

---

## Results

### Performance Metrics

The model is evaluated using:

- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Precision**: Ratio of correct predictions
- **Recall**: Ratio of detected objects
- **Per-class metrics**: Individual letter performance

### Expected Performance

With 50 epochs of training on YOLOv8n:
- **mAP50**: ~0.85-0.95
- **Inference Speed**: 20-30 FPS (GPU) / 2-5 FPS (CPU)
- **Training Time**: 30-60 minutes (GPU) / 4-8 hours (CPU)

---

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions

- Add support for ASL words/phrases
- Implement model quantization for mobile deployment
- Create web interface using Flask/Streamlit
- Add data augmentation techniques
- Improve real-time detection accuracy
- Create Jupyter notebooks for tutorials

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Dataset License:** The American Sign Language Letters dataset is licensed under Public Domain by David Lee.

---

## Acknowledgments

- **Dataset**: [David Lee](https://www.linkedin.com/in/daviddaeshinlee/) for creating and sharing the ASL dataset
- **Framework**: [Ultralytics](https://github.com/ultralytics/ultralytics) for the excellent YOLOv8/YOLO11 framework
- **Platform**: [Roboflow Universe](https://universe.roboflow.com/) for hosting the dataset
- **Inspiration**: Building accessible technology for the deaf and hard-of-hearing community

---

## Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [ASL Dataset on Roboflow](https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters)
- [Computer Vision ASL Blog Post](https://blog.roboflow.com/computer-vision-american-sign-language/)

---

## Contact & Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check [SETUP.md](SETUP.md) or [examples/quick_start.md](examples/quick_start.md)
- **Documentation**: See [docs/](docs/) for additional guides

---

**Made with ❤️ for learning computer vision and supporting accessibility**
