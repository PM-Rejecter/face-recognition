import glob
from DataBase.DataSummoner import DataSummoner
from pathlib import Path
import cv2
import json
ds = DataSummoner("../DataBase/exodia-face-recognition-firebase-adminsdk-1xywr-4f1f07fee9.json")

def run():
    path = Path.cwd() / "origin_images"
    output_path = Path.cwd() / "processed_images"
    path_str = str(path)
    if not path.exists():
        ds.download_all_images(path_str)
    all_images = str(path /"*")
    for path in glob.glob(all_images,recursive=True):
        with open(path+'/image_info.json', newline='') as jsonfile:
            data_info = json.load(jsonfile)
        images = []
        for image_path in glob.glob(path + "/*.jpg"):
            img = cv2.imread(image_path)
            # 1. get image landmark
            # 2. rotate_img
            # 3. rebuild bounding box
            # 4. check and resize bounding box
            # 5. cut bounding box and download images

            images.append(ds.image_to_base64(img))  # processed image and add input image type
        data_info['is_processed'] = True
        data_info['data'] = images
        ds.save_data(data=data_info)


# TODO:Flow: preprocess
"""
1. load images
2. get image landmark
3. rotate_img 
4. rebuild bounding box
5. check and resize bounding box
6. cut bounding box and download images
"""






if __name__=="__main__":
    run()