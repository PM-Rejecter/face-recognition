import cv2
from pathlib import Path
import dlib
import numpy as np
from enum import Enum

class DetectorType(Enum):
    Cascade = 1
    HOG_LinearSVM = 2

class LandmarksPredictorType(Enum):
    EnsembleRegressionTrees = 1

class FaceLandmarks:
    def __init__(self, bbox_detector=DetectorType.Cascade, landmarks_model=LandmarksPredictorType.EnsembleRegressionTrees):

        self.bbox_detector = bbox_detector
        self.landmarks_model = landmarks_model

    def bounding_box(self, img) -> np.array:
        if self.bbox_detector == DetectorType.Cascade:
            model_path = Path('../models/haarcascade_frontalface_default.xml').resolve()
            detector = cv2.CascadeClassifier(str(model_path))
            bbox = detector.detectMultiScale(img)
        elif self.bbox_detector == DetectorType.HOG_LinearSVM:
            detector = dlib.get_frontal_face_detector()
            dlib_bbox = detector(img, 1) # up-sampling once
            bbox = []
            for face in dlib_bbox:
                x, y, width, height = face.left(), face.top(), face.right() - face.left(), face.right() - face.left()
                bbox.append([x, y, width, height])
            bbox = np.array(bbox)
        return bbox

    def facial_landmarks(self, img) -> dlib.points:
        bbox = self.bounding_box(img)
        if self.landmarks_model == LandmarksPredictorType.EnsembleRegressionTrees:
            model_path = Path('../models/shape_predictor_68_face_landmarks.dat').resolve()
            predictor = dlib.shape_predictor(str(model_path))
            landmarks = []
            for face in bbox:
                x1, y1, width, height = face
                x2, y2 = x1 + width, y1 + height
                left = x1
                right = x2
                top = y1
                bottom = y2
                dlib_rect = dlib.rectangle(left, top, right, bottom)
                points = predictor(img, dlib_rect).parts()
                landmarks.append(points)
        return landmarks

    def alignment_points(self, img) -> np.array:
        landmarks = self.facial_landmarks(img)
        alignment_points = []
        for points in landmarks:
            eye1 = np.mean([[points[36].x, points[36].y], [points[39].x, points[39].y]], axis=0)
            eye2 = np.mean([[points[42].x, points[42].y], [points[45].x, points[45].y]], axis=0)
            nose = np.array([points[33].x, points[33].y])
            alignment_points.append([eye1, eye2, nose])
        alignment_points = np.array(alignment_points)
        return alignment_points # 0:left eye, 1:right eye, 2:nose

    def plotting(self, img):
        bbox = self.bounding_box(img)
        landmarks = self.facial_landmarks(img)
        for face in bbox:
            x1, y1, width, height = face
            x2, y2 = x1 + width, y1 + height
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 2)
        for points in landmarks:
            for p in points:
                x = p.x
                y = p.y
                cv2.circle(img, (x, y), 2, (0, 0, 200), cv2.FILLED)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    def plotting_face_alignment_points(self, img):
        bbox = self.bounding_box(img)
        features = self.alignment_points(img)
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