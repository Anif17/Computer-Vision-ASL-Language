"""
YOLOv8 Training Script for ASL Alphabet Recognition
This script trains a YOLOv8 model to detect and classify American Sign Language letters.
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_yolo_asl():
    """
    Train YOLOv8 model on ASL alphabet dataset
    """

    # Set paths (go up one level from scripts/ to project root)
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "Dataset ASL Languauge"
    data_yaml = dataset_path / "data.yaml"

    print("=" * 60)
    print("YOLOv8 ASL Alphabet Training")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Data config: {data_yaml}")

    # Check if dataset exists
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration not found at {data_yaml}")

    # Model options:
    # yolov8n.pt - Nano (fastest, smallest)
    # yolov8s.pt - Small
    # yolov8m.pt - Medium
    # yolov8l.pt - Large
    # yolov8x.pt - Extra Large (best accuracy, slowest)

    model_size = "yolov8n.pt"  # Start with nano for faster training
    print(f"\nLoading model: {model_size}")

    # Load a pretrained YOLOv8 model
    model = YOLO(model_size)

    # Training parameters
    print("\nTraining Configuration:")
    print("-" * 60)

    training_params = {
        'data': str(data_yaml),           # Path to data.yaml
        'epochs': 50,                      # Number of training epochs
        'imgsz': 640,                      # Input image size
        'batch': 16,                       # Batch size (adjust based on GPU memory)
        'patience': 10,                    # Early stopping patience
        'save': True,                      # Save checkpoints
        'device': 0,                       # GPU device (0 for first GPU, 'cpu' for CPU)
        'workers': 4,                      # Number of worker threads for data loading
        'project': 'runs/asl_detection',   # Project directory
        'name': 'yolov8_asl_training',     # Experiment name
        'exist_ok': True,                  # Overwrite existing experiment
        'pretrained': True,                # Use pretrained weights
        'optimizer': 'auto',               # Optimizer (auto, SGD, Adam, AdamW, etc.)
        'verbose': True,                   # Verbose output
        'seed': 42,                        # Random seed for reproducibility
        'deterministic': True,             # Deterministic mode
        'single_cls': False,               # Train as single-class dataset
        'rect': False,                     # Rectangular training
        'cos_lr': False,                   # Cosine learning rate scheduler
        'close_mosaic': 10,                # Disable mosaic augmentation for final epochs
        'resume': False,                   # Resume training from last checkpoint
        'amp': True,                       # Automatic Mixed Precision training
        'fraction': 1.0,                   # Fraction of dataset to use
        'profile': False,                  # Profile ONNX and TensorRT speeds
        'freeze': None,                    # Freeze layers (e.g., [0, 1, 2])
        'lr0': 0.01,                       # Initial learning rate
        'lrf': 0.01,                       # Final learning rate (lr0 * lrf)
        'momentum': 0.937,                 # SGD momentum/Adam beta1
        'weight_decay': 0.0005,            # Optimizer weight decay
        'warmup_epochs': 3.0,              # Warmup epochs
        'warmup_momentum': 0.8,            # Warmup initial momentum
        'warmup_bias_lr': 0.1,             # Warmup initial bias learning rate
        'box': 7.5,                        # Box loss gain
        'cls': 0.5,                        # Class loss gain
        'dfl': 1.5,                        # DFL loss gain
        'pose': 12.0,                      # Pose loss gain (for pose models)
        'kobj': 2.0,                       # Keypoint obj loss gain (for pose models)
        'label_smoothing': 0.0,            # Label smoothing epsilon
        'nbs': 64,                         # Nominal batch size
        'hsv_h': 0.015,                    # HSV-Hue augmentation
        'hsv_s': 0.7,                      # HSV-Saturation augmentation
        'hsv_v': 0.4,                      # HSV-Value augmentation
        'degrees': 0.0,                    # Rotation augmentation (degrees)
        'translate': 0.1,                  # Translation augmentation (fraction)
        'scale': 0.5,                      # Scale augmentation (gain)
        'shear': 0.0,                      # Shear augmentation (degrees)
        'perspective': 0.0,                # Perspective augmentation (gain)
        'flipud': 0.0,                     # Vertical flip augmentation (probability)
        'fliplr': 0.5,                     # Horizontal flip augmentation (probability)
        'mosaic': 1.0,                     # Mosaic augmentation (probability)
        'mixup': 0.0,                      # Mixup augmentation (probability)
        'copy_paste': 0.0,                 # Copy-paste augmentation (probability)
    }

    for key, value in training_params.items():
        print(f"{key:20s}: {value}")

    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")

    # Train the model
    try:
        results = model.train(**training_params)

        print("\n" + "=" * 60)
        print("Training Completed Successfully!")
        print("=" * 60)
        print(f"\nBest model saved at: runs/asl_detection/yolov8_asl_training/weights/best.pt")
        print(f"Last model saved at: runs/asl_detection/yolov8_asl_training/weights/last.pt")

        # Validate the model
        print("\n" + "=" * 60)
        print("Running Validation...")
        print("=" * 60 + "\n")

        metrics = model.val()

        print("\nValidation Metrics:")
        print("-" * 60)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")

        return results, metrics

    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOLOv8 ASL Alphabet Detection - Training Script")
    print("=" * 60 + "\n")

    try:
        results, metrics = train_yolo_asl()
        print("\n" + "=" * 60)
        print("Training pipeline completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")
        raise
