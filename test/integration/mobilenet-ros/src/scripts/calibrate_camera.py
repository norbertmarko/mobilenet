import numpy as np
import cv2

class CalibrateCamera:
    
    def __init__(self, calibration_images, pattern_size=(9, 6), retain_calibration_images=False):

        """
        Image warping during bird's eye view transformations
        would make camera distortions apparent (inaccurate predictions).
        SOLUTION: Shoot some images with the camera used for the task
        and calibrate it with this pipeline.
        
        Parameters
        -----------------
        calibration_images       : Calibration Images.
        pattern_size             : Shape of calibration pattern.
        retain_calibration_images: Flag indicating if we need to preserve calibration images.
        """
        
        pass

    def calculate(images, pattern_size, retain_calibration_images):

        pattern = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)

    