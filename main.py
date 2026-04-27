import cv2
import argparse
import os
from src.camera import CameraService
from src.detector import ArucoDetector

WINDOW_NAME = "ArUconator"

def main():
    parser = argparse.ArgumentParser(description="ArUco Screen Replacement")
    parser.add_argument("--input", type=str, help="Path to input video file. If not provided, camera will be used.")
    parser.add_argument("--output", type=str, help="Path to save the processed video.")
    parser.add_argument("--overlay", type=str, default="./data/image.png", help="Path to the overlay image.")
    args = parser.parse_args()

    if args.input:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.input}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        camera_service = CameraService(camera_index=2)
        cap = camera_service.cap
        fps = 30.0
        width = 1280
        height = 720

    detector = ArucoDetector(overlay_image_path=args.overlay)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to {args.output}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)

    print("Press Escape or close the window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = detector.process_frame(frame)

            if writer:
                writer.write(processed_frame)

            cv2.imshow(WINDOW_NAME, processed_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Escape
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()