from pathlib import Path

import cv2


class FaceDetector:
    def __init__(self, model='CascadeClassifier'):
        if model == 'CascadeClassifier':
            model_path = Path('models/haarcascade_frontalface_default.xml').resolve()
            self.classifier = cv2.CascadeClassifier(str(model_path))
        else:
            raise NameError('Model is not exist.')

    def _bounding_box(self):
        pass

    def video_capture(self, camera_index=0):
        classifier = self.classifier

        cap = cv2.VideoCapture(camera_index)

        while True:
            ret, img = cap.read()
            bbox = classifier.detectMultiScale(img)

            for box in bbox:
                x, y, width, height = box
                x2, y2 = x + width, y + height
                cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 5)

            cv2.imshow("face check", img)

            if cv2.waitKey(30) & 0xff == 27:
                break

            cv2.destroyWindow('face check')