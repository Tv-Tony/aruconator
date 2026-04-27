import cv2

class CameraService:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

    def get_frame(self):
        # Grabs a single picture from the video stream
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def close(self):
        # Cleans up the hardware
        self.cap.release()