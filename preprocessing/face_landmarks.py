import cv2
from pathlib import Path
import dlib
import numpy as np

class FaceLandmarks:
    def __init__(self, bbox_detector='CascadeClassifier', landmarks_model='EnsembleRegressionTrees'):
        if bbox_detector == 'HOG_LinearSVM':
            self.detector = dlib.get_frontal_face_detector()
        elif bbox_detector == 'CascadeClassifier':
            model_path = Path('../models/haarcascade_frontalface_default.xml').resolve()
            self.detector = cv2.CascadeClassifier(str(model_path))
        else:
            raise NameError('Bounding box model is not existing.')

        if landmarks_model == 'EnsembleRegressionTrees':
            model_path = Path('../models/shape_predictor_68_face_landmarks.dat').resolve()
            self.predictor = dlib.shape_predictor(str(model_path))
        else:
            raise NameError('Landmarks model is not existing.')

    def _get_bounding_box(self, img):
        bbox = self.detector.detectMultiScale(img)
        return bbox

    def _get_facial_landmarks(self, img):
        bbox = self._get_bounding_box(img)
        landmarks = []
        for face in bbox:
            x1, y1, width, height = face
            x2, y2 = x1 + width, y1 + height
            left = x1
            right = x2
            top = y1
            bottom = y2
            dlibRect = dlib.rectangle(left, top, right, bottom)
            points = self.predictor(img, dlibRect).parts()
            landmarks.append(points)
        return landmarks

    def _get_alignment_points(self, img):
        landmarks = self._get_facial_landmarks(img)
        alignment_points = []
        for points in landmarks:
            eye1 = np.mean([np.array([points[36].x, points[36].y]), np.array([points[39].x, points[39].y])], axis=0)
            eye2 = np.mean([np.array([points[42].x, points[42].y]), np.array([points[45].x, points[45].y])], axis=0)
            nose1 = np.array([points[31].x, points[31].y])
            nose2 = np.array([points[33].x, points[33].y])
            nose3 = np.array([points[35].x, points[35].y])
            alignment_points.append([eye1, eye2, nose1, nose2, nose3])
        return alignment_points

    def plotting(self, img):
        bbox = self._get_bounding_box(img)
        landmarks = self._get_facial_landmarks(img, bbox)
        for face in bbox:
            x1, y1, width, height = face
            x2, y2 = x1 + width, y1 + height
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 180), 2)
        for points in landmarks:
            for p in points:
                x = p.x
                y = p.y
                cv2.circle(img, (x, y), 5, (50, 255, 255), cv2.FILLED)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    def plotting_face_alignment_points(self, img):
        bbox = self._get_bounding_box(img)
        features = self._get_alignment_points(img)
        for face in bbox:
            x1, y1, width, height = face
            x2, y2 = x1 + width, y1 + height
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 180), 2)
        for face in features:
            for p in face:
                p_ = dlib.point((p[0], p[1]))
                x, y = p_.x, p_.y
                cv2.circle(img, (x, y), 3, (50, 50, 255), cv2.FILLED)
        cv2.imshow('image', img)
        cv2.waitKey(0)

basketball = cv2.imread(r'C:\Users\User\Desktop\try.jpeg')
temp1 = FaceLandmarks()._get_bounding_box(basketball)
temp2 = FaceLandmarks()._get_facial_landmarks(basketball)
temp3 = FaceLandmarks()._get_alignment_points(basketball)
FaceLandmarks().plotting_face_alignment_points(basketball)
FaceLandmarks().plotting(basketball)
