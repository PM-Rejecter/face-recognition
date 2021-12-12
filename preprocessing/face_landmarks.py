import cv2
from pathlib import Path
import dlib
import numpy as np

class FaceLandmarks:
    def __init__(self,img,  bbox_detector = 'HOG_LinearSVM', landmarks_model = 'EnsembleRegressionTrees'):
        if bbox_detector == 'HOG_LinearSVM':
            self.detector = dlib.get_frontal_face_detector()
        else:
            raise NameError('Bounding box model is not existing.')

        if landmarks_model == 'EnsembleRegressionTrees':
            model_path = Path('../models/shape_predictor_68_face_landmarks.dat').resolve()
            self.predictor = dlib.shape_predictor(str(model_path))
        else:
            raise NameError('Landmarks model is not existing.')

        self.img = img

    def _get_bounding_box(self):
        faces = self.detector(self.img)
        return faces

    def _get_facial_landmarks(self):
        faces = self._get_bounding_box()
        landmarks = []
        for face in faces:
            points = self.predictor(self.img, face).parts()
            landmarks.append(points)
        return landmarks

    def _get_alignment_points(self):
        landmarks = self._get_facial_landmarks()
        for face in landmarks:
            eye1 = np.matrix([(p.x, p.y) for p in face[36:42]]).mean(axis=0)
            eye2 = np.matrix([(p.x, p.y) for p in face[42:48]]).mean(axis=0)
            nose1 = np.matrix([face[31].x, face[31].y])
            nose2 = np.matrix([face[33].x, face[33].y])
            nose3 = np.matrix([face[35].x, face[35].y])
        alignment_points = [eye1, eye2, nose1, nose2, nose3]
        return alignment_points

    def _plotting(self):
        faces = self._get_bounding_box()
        landmarks = self._get_facial_landmarks()
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            imgShow = cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for points in landmarks:
            for p in points:
                x = p.x
                y = p.y
                imgShow = cv2.circle(imgShow, (x, y), 3, (50, 50, 255), cv2.FILLED)
        cv2.imshow('image', imgShow)
        cv2.waitKey(0)


Thea = cv2.imread(r'C:\Users\User\Desktop\Thea.jpg')
Thea1 = cv2.imread(r'C:\Users\User\Desktop\Thea1.jpg')
basketball = cv2.imread(r'C:\Users\User\Desktop\try.jpeg')
Jude = cv2.imread(r'C:\Users\User\Desktop\Jude.jpg')
x = FaceLandmarks(basketball)