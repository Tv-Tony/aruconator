import cv2
import numpy as np
from src.transform import get_perspective_transform, custom_warp_perspective

class ArucoDetector:
    def __init__(self, overlay_image_path="./data/image.png"):

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        # Load the overlay image
        self.overlay_image = cv2.imread(overlay_image_path)
        if self.overlay_image is None:
            print(f"Warning: Could not load overlay image from '{overlay_image_path}'. Overlay will be skipped.")

    def _get_marker_center(self, corners):
        return corners[0].mean(axis=0)

    def process_frame(self, frame):
        if frame is None:
            return frame

        corners, ids, rejected = self.detector.detectMarkers(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if self.overlay_image is not None:
                frame = self._apply_overlay(frame, corners, ids)

        return frame

    def _apply_overlay(self, frame, corners, ids):
     
        if len(ids) < 4:
            return frame  

        # Get centers the 4 markers
        marker_centers = []
        for i in range(4):
            marker_centers.append(self._get_marker_center(corners[i]))
        
        marker_centers = np.array(marker_centers)

        sorted_by_y = marker_centers[np.argsort(marker_centers[:, 1])]
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]

        tl = top_two[np.argsort(top_two[:, 0])[0]]
        tr = top_two[np.argsort(top_two[:, 0])[1]]

        bl = bottom_two[np.argsort(bottom_two[:, 0])[0]]
        br = bottom_two[np.argsort(bottom_two[:, 0])[1]]

        dst_points = np.array([tl, tr, br, bl], dtype=np.float32)

        h, w = self.overlay_image.shape[:2]
        src_points = np.array([
            [0,   0  ],
            [w,   0  ],
            [w,   h  ],
            [0,   h  ],
        ], dtype=np.float32)

        homography_matrix = get_perspective_transform(src_points, dst_points)
        
        frame_h, frame_w = frame.shape[:2]
        warped = custom_warp_perspective(self.overlay_image, homography_matrix, (frame_w, frame_h))

        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_points.astype(np.int32), 255)

        frame_masked = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

        warped_masked = cv2.bitwise_and(warped, warped, mask=mask)

        result = cv2.add(frame_masked, warped_masked)
        return result