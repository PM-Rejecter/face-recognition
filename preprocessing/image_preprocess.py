import glob
from DataBase.DataSummoner import DataSummoner
from pathlib import Path
import cv2
import json
from face_landmarks import FaceLandmarks
from image_adjuster import rotate_img
import numpy as np


def preprocess(image_path: str, key_path: str, output_path: str = None):
    ds = DataSummoner(key_path)
    path = Path(image_path)
    # output_path = Path(output_path) if output_path else Path.cwd() / "processed_images"
    face_landmarks = FaceLandmarks()
    path_str = str(path)
    if not path.exists():
        ds.download_all_images(path_str)
    all_images = str(path / "*")
    for path in glob.glob(all_images, recursive=True):
        with open(path + '/image_info.json', newline='') as jsonfile:
            data_info = json.load(jsonfile)
        images = []
        bounding_boxes = []
        for image_path in glob.glob(path + "/*.jpg"):
            img = cv2.imread(image_path)
            land_marks = face_landmarks.alignment_points(img)
            right_eye = land_marks[1]
            left_eye = land_marks[0]
            center_x = (right_eye[0] + left_eye[0]) / 2
            center_y = (right_eye[1] + left_eye[1]) / 2
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            rotated_img = rotate_img(img, [center_x, center_y], angle)
            bounding_box = face_landmarks.bounding_box(rotated_img)
            crop_img = rotated_img[bounding_box[1]: bounding_box[1] + bounding_box[2],
                       bounding_box[0]:bounding_box[0] + bounding_box[3]]
            bounding_boxes.append(bounding_box.tolist())
            images.append(ds.image_to_base64(crop_img, 'image'))  # processed image and add input image type
        data_info['is_processed'] = True
        data_info['data'] = images
        data_info['bounding_boxes'] = {str(i): b for i, b in enumerate(bounding_boxes)}
        ds.save_data(data=data_info)

