import cv2
#import matplotlib.pyplot as plt
import dlib
# pip install cmake
import numpy as np
from preprocessing import FaceDetector

#detector.video_capture()

'''
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/PM-Rejecter/Face-Recognition/shape_predictor_68_face_landmarks.dat')
img = cv2.imread('C:/Users/User/Desktop/image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(img_gray, 0)
if len(faces) != 0:
    for i in range(len(faces)):
        landmarks = np.matrix([p.x, p.y] for p in predictor(img, faces[i]).parts())
        for idx, point in enumerate(landmarks):
            pos = (point([0,0], point[0,1])) #68個座標點
'''


class FaceLandmarks:
    def __init__(self, detector = FaceDetector(), predictor_path:str):
        self.predictor = dlib.shape_predictor(predictor_path)

    def face_landmarks(self):

        # 偵測到人臉
        if len(faces) != 0:
            # 取特徵點座標
            for i in range(len(faces)):
                landmarks = np.matrix([p.x, p.y] for p in predictor(img, faces[i]).parts())
                for idx, point in enumerate(landmarks):
                    pos = (point([0, 0], point[0, 1]))  # 68個座標點