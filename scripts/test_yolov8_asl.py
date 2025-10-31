"""
YOLOv8 Testing/Inference Script for ASL Alphabet Recognition
This script tests a trained YOLOv8 model on images or test dataset.
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse

def test_on_image(model_path, image_path, conf_threshold=0.25, save_output=True):
    """
    Run inference on a single image

    Args:
        model_path (str): Path to trained model weights
        image_path (str): Path to test image
        conf_threshold (float): Confidence threshold for detections
        save_output (bool): Whether to save output image
    """
    print("=" * 60)
    print("YOLOv8 ASL - Single Image Inference")
    print("=" * 60)

    # Load the model
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")

    # Run inference
    results = model(image_path, conf=conf_threshold)

    # Process results
    for result in results:
        # Get the annotated image
        annotated_img = result.plot()

        # Display results
        print(f"\nDetections found: {len(result.boxes)}")

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            print(f"  - Letter: {cls_name}, Confidence: {conf:.2f}")

        # Save or display the image
        if save_output:
            output_path = Path("outputs") / f"result_{Path(image_path).name}"
            output_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output_path), annotated_img)
            print(f"\nOutput saved to: {output_path}")

        # Display the image
        cv2.imshow("YOLOv8 ASL Detection", annotated_img)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_on_dataset(model_path, data_yaml):
    """
    Run inference on entire test dataset

    Args:
        model_path (str): Path to trained model weights
        data_yaml (str): Path to data.yaml configuration
    """
    print("=" * 60)
    print("YOLOv8 ASL - Test Dataset Evaluation")
    print("=" * 60)

    # Load the model
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")

    # Validate on test set
    print("\nRunning validation on test set...")
    metrics = model.val(data=data_yaml, split='test')

    print("\n" + "=" * 60)
    print("Test Set Metrics:")
    print("=" * 60)
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")

    # Per-class metrics
    print("\nPer-Class mAP50:")
    print("-" * 60)
    for i, class_name in enumerate(model.names.values()):
        if i < len(metrics.box.maps):
            print(f"{class_name:5s}: {metrics.box.maps[i]:.4f}")

    return metrics

def batch_inference(model_path, image_folder, conf_threshold=0.25, save_output=True):
    """
    Run inference on multiple images in a folder

    Args:
        model_path (str): Path to trained model weights
        image_folder (str): Path to folder containing images
        conf_threshold (float): Confidence threshold for detections
        save_output (bool): Whether to save output images
    """
    print("=" * 60)
    print("YOLOv8 ASL - Batch Inference")
    print("=" * 60)

    # Load the model
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")

    # Get all image files
    image_folder = Path(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in image_folder.iterdir()
                   if f.suffix.lower() in image_extensions]

    print(f"\nFound {len(image_files)} images in {image_folder}")

    output_dir = Path("outputs/batch_results")
    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"\nProcessing [{i}/{len(image_files)}]: {img_path.name}")

        # Run inference
        results = model(str(img_path), conf=conf_threshold)

        for result in results:
            print(f"  Detections: {len(result.boxes)}")

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                print(f"    - {cls_name}: {conf:.2f}")

            if save_output:
                annotated_img = result.plot()
                output_path = output_dir / f"result_{img_path.name}"
                cv2.imwrite(str(output_path), annotated_img)

    if save_output:
        print(f"\nAll results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 ASL Alphabet Testing')
    parser.add_argument('--model', type=str,
                       default='runs/asl_detection/yolov8_asl_training/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--mode', type=str, choices=['image', 'dataset', 'batch'],
                       default='dataset',
                       help='Testing mode: image, dataset, or batch')
    parser.add_argument('--source', type=str,
                       default='Dataset ASL Languauge/data.yaml',
                       help='Image path, data.yaml path, or folder path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save output images')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("YOLOv8 ASL Alphabet Detection - Testing Script")
    print("=" * 60 + "\n")

    try:
        if args.mode == 'image':
            test_on_image(args.model, args.source, args.conf, args.save)
        elif args.mode == 'dataset':
            test_on_dataset(args.model, args.source)
        elif args.mode == 'batch':
            batch_inference(args.model, args.source, args.conf, args.save)

        print("\n" + "=" * 60)
        print("Testing completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
