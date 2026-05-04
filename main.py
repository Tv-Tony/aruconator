import cv2
import argparse
from pathlib import Path
from src.detector import ArucoDetector
 
videos = [("horizontal.mp4", "horizontal.mp4"), ("vertical.mp4", "vertical.mp4")]
 
data_dir = Path("data/video_input")
output_dir = Path("data/video_output")
 
 
def process_video(input_path: Path, output_path: Path, overlay_image_path: str):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
 
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    detector = ArucoDetector(overlay_image_path=overlay_image_path)
 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
 
    print(f"Processing: {input_path} -> {output_path}")
 
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
 
            processed_frame = detector.process_frame(frame)
            writer.write(processed_frame)
            frame_count += 1
 
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
    finally:
        cap.release()
        writer.release()
 
    print(f"  Done. {frame_count} frames written to {output_path}")
 
 
def main():
    parser = argparse.ArgumentParser(description="ArUco Screen Replacement")
    parser.add_argument(
        "--overlay",
        type=str,
        default="./data/image.png",
        help="Path to the overlay image.",
    )
    args = parser.parse_args()
 
    output_dir.mkdir(parents=True, exist_ok=True)
 
    for input_filename, output_filename in videos:
        input_path = data_dir / input_filename
        output_path = output_dir / output_filename
 
        if not input_path.exists():
            print(f"Warning: Input file not found, skipping: {input_path}")
            continue
 
        process_video(input_path, output_path, args.overlay)
 
    print("All videos processed.")
 
 
if __name__ == "__main__":
    main()
 
