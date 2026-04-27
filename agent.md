# ArUconator Project - AI Agent Instructions

## Project Overview
This project is an Augmented Reality (AR) Python application. It uses OpenCV to detect ArUco markers through a live video feed and overlays a custom image (`/data/image.png`) bounded by four specific markers.

## Tech Stack & Tooling
* **Package Manager:** `uv`
* **Language:** Python
* **Computer Vision:** `opencv-contrib-python`, `numpy`
* **UI/Web (Planned):** `gradio`
* **Environment:** Linux (Video4Linux2 backend required for camera)

## Architecture Rules (OOP Strict)
The project strictly follows Object-Oriented Programming (OOP) and Separation of Concerns. Do not put all logic into one file.
* `src/camera.py`: Only handles hardware interaction (cv2.VideoCapture using `cv2.CAP_V4L2`).
* `src/detector.py`: Only handles CV2 math, ArUco detection, and image warping.
* `main.py`: The entry point. Connects the camera to the detector and handles the display window.

## Current State
* ArUco markers (DICT_6X6_50) are successfully being detected.
* Camera feed is functional via native Linux `/dev/video*` devices.

---

## New Feature Implementation: Image Overlay (Homography)

### Objective
Map a custom image (`/data/image.png`) onto the quadrilateral defined by 4 ArUco markers in the physical world. The image must transform in real-time as the camera moves.

### Requirements & Marker Mapping
1.  **Markers:** The user will place exactly four markers to act as the corners of the bounding box.
    * ID 0: Top-Left
    * ID 1: Top-Right
    * ID 2: Bottom-Right
    * ID 3: Bottom-Left
2.  **Asset:** Load the image from `./data/image.png`.

### Technical Steps (for `src/detector.py`)
1.  **Extract Centers:** When markers 0, 1, 2, and 3 are detected, calculate the center (X, Y) coordinate for each marker (by averaging its 4 corners).
2.  **Source Coordinates:** Define the 4 corners of the source image (`/data/image.png`):
    * `[0, 0]`
    * `[image_width, 0]`
    * `[image_width, image_height]`
    * `[0, image_height]`
3.  **Destination Coordinates:** Map the source corners to the detected marker centers (in the order: 0, 1, 2, 3).
4.  **Calculate Homography:** Use `cv2.findHomography(src_points, dst_points)` to generate the transformation matrix.
5.  **Warp Image:** Use `cv2.warpPerspective()` to stretch the source image to match the camera frame dimensions based on the homography matrix.
6.  **Overlay (Masking):** * Create a black polygon mask on the original frame using the destination coordinates (`cv2.fillConvexPoly`).
    * Combine the original frame and the warped image seamlessly using `cv2.add()`.
    