"""
YOLOv8 Real-Time Webcam Detection for ASL Alphabet Recognition
This script performs real-time ASL letter detection using your webcam.
"""

from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path
import time

class ASLWebcamDetector:
    """Real-time ASL letter detector using webcam"""

    def __init__(self, model_path, conf_threshold=0.5, camera_id=0):
        """
        Initialize the detector

        Args:
            model_path (str): Path to trained YOLO model
            conf_threshold (float): Confidence threshold for detections
            camera_id (int): Camera device ID (0 for default webcam)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.camera_id = camera_id
        self.cap = None

        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.detected_letters = []
        self.letter_history = []  # Track last N detections
        self.history_length = 10

        print("=" * 60)
        print("YOLOv8 ASL Real-Time Detection Initialized")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Confidence Threshold: {conf_threshold}")
        print(f"Camera ID: {camera_id}")
        print("=" * 60)

    def start_camera(self):
        """Initialize the webcam"""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("\nCamera started successfully!")
        print("\nControls:")
        print("  - Press 'q' or 'ESC' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'c' to clear detected letters")
        print("  - Press '+' to increase confidence threshold")
        print("  - Press '-' to decrease confidence threshold")
        print("=" * 60 + "\n")

    def process_frame(self, frame):
        """
        Process a single frame with YOLO detection

        Args:
            frame: Input frame from webcam

        Returns:
            Annotated frame with detections
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        # Get the annotated frame
        annotated_frame = results[0].plot()

        # Extract detections
        current_detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.model.names[cls_id]
            current_detections.append((cls_name, conf))

        # Update letter history
        if current_detections:
            # Get the highest confidence detection
            best_detection = max(current_detections, key=lambda x: x[1])
            self.letter_history.append(best_detection[0])

            # Keep only last N detections
            if len(self.letter_history) > self.history_length:
                self.letter_history.pop(0)

            # If we have a stable detection (same letter appears frequently)
            if len(self.letter_history) >= 5:
                most_common = max(set(self.letter_history), key=self.letter_history.count)
                if self.letter_history.count(most_common) >= 5:
                    if not self.detected_letters or self.detected_letters[-1] != most_common:
                        self.detected_letters.append(most_common)

        return annotated_frame, current_detections

    def draw_info_panel(self, frame, detections, start_time):
        """
        Draw information panel on the frame

        Args:
            frame: Frame to draw on
            detections: List of (letter, confidence) tuples
            start_time: Time when processing started

        Returns:
            Frame with info panel
        """
        height, width = frame.shape[:2]

        # Calculate FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            self.fps = 1 / elapsed_time

        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        # Draw confidence threshold
        conf_text = f"Conf: {self.conf_threshold:.2f}"
        cv2.putText(frame, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        # Draw detections count
        det_text = f"Detections: {len(detections)}"
        cv2.putText(frame, det_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        # Draw detected letters string
        if self.detected_letters:
            letters_text = "Letters: " + "".join(self.detected_letters[-15:])  # Show last 15
            cv2.putText(frame, letters_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 0), 2)

        # Draw current detections
        if detections:
            current_text = f"Current: {detections[0][0]} ({detections[0][1]:.2f})"
            cv2.putText(frame, current_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)

        return frame

    def run(self):
        """Main detection loop"""
        self.start_camera()

        try:
            while True:
                start_time = time.time()

                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                # Process frame
                annotated_frame, detections = self.process_frame(frame)

                # Add info panel
                annotated_frame = self.draw_info_panel(annotated_frame, detections, start_time)

                # Display frame
                cv2.imshow("YOLOv8 ASL Real-Time Detection", annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nExiting...")
                    break

                elif key == ord('s'):  # Save frame
                    output_dir = Path("outputs/webcam_captures")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = output_dir / f"capture_{timestamp}.jpg"
                    cv2.imwrite(str(filename), annotated_frame)
                    print(f"Frame saved: {filename}")

                elif key == ord('c'):  # Clear detected letters
                    self.detected_letters.clear()
                    self.letter_history.clear()
                    print("Detected letters cleared")

                elif key == ord('+') or key == ord('='):  # Increase confidence
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"Confidence threshold: {self.conf_threshold:.2f}")

                elif key == ord('-'):  # Decrease confidence
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"Confidence threshold: {self.conf_threshold:.2f}")

                self.frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("Session Summary")
        print("=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Average FPS: {self.fps:.1f}")
        if self.detected_letters:
            print(f"Detected letters: {''.join(self.detected_letters)}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 ASL Real-Time Webcam Detection')
    parser.add_argument('--model', type=str,
                       default='runs/asl_detection/yolov8_asl_training/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        print("Please train a model first using train_yolov8_asl.py")
        return

    try:
        detector = ASLWebcamDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            camera_id=args.camera
        )
        detector.run()

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
