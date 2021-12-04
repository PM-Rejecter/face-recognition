import numpy as np
import cv2


class FaceAligner:
    def __init__(self, landmarks_predictor, left_eye_position=(0.35, 0.35), image_size=(256, 256)):
        self.landmarks_predictor = landmarks_predictor
        self.left_eye_position = left_eye_position
        self.width, self.height = image_size

    def align(self, image, rect):
        pass
