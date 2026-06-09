import cv2
import numpy as np

class ScaleCalibrator:
    def __init__(self, aruco_dict_type=cv2.aruco.DICT_5X5_50, marker_size_mm=50.0):
        """
        Initializes the Aruco detector.
        Default is a 50mm 5x5 ArUco marker.
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size_mm = marker_size_mm

    def calibrate(self, image):
        """
        Detects an ArUco marker in the image and calculates the scale factor (mm/px).
        Returns the scale factor, or None if no marker is found.
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            # For simplicity, we use the first detected marker
            c = corners[0][0] # c shape is (4, 2)
            
            # Calculate perimeter in pixels
            perim_px = cv2.arcLength(c, True)
            
            # The perimeter of the marker in mm is 4 * marker_size_mm
            perim_mm = 4.0 * self.marker_size_mm
            
            # Scale factor: mm / px
            scale_factor = perim_mm / perim_px
            
            return scale_factor, corners[0]
            
        return None, None
