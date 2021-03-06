import base64
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from PIL import Image
from io import BytesIO
from typing import Dict
from pathlib import Path
import uuid
import json


class DataSummoner:
    def __init__(self, cred_path: str):
        # path/to/serviceAccoun t.json 請用自己存放的路徑
        cred = credentials.Certificate(cred_path)
        # 初始 化firebase，注意不能重複初始化
        firebase_admin.initialize_app(cred)
        # 初始化firestore
        self.image_set = 'face_images'
        self.db = firestore.client()
        self.columns_dict = {'face_images': ['data', 'name', 'gender', 'is_processed', 'image_id', 'bounding_boxes']}

    def base64_to_image(self, base64_str):
        try:
            im = Image.open(BytesIO(base64.b64decode(base64_str)))
            return im
        except Exception as e:
            print(e)

    def image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            return encoded_string
        except Exception as e:
            print(e)

    def save_data(self, data: Dict, id: str = None):
        try:
            photo_ref = self.db.collection(self.image_set)
            if sorted(list(data.keys())) == sorted(self.columns_dict[self.image_set]):
                if id:
                    photo_ref.document(id).set(data)
                else:
                    photo_ref.add(data)
            else:
                print('save fail')
                print('[ERROR]->columns is not match')
        except Exception as e:
            print(e)

    def delete_data(self, document: str):
        try:
            self.db.collection(self.image_set).document(document).delete()
            print("done!")
        except Exception as e:
            print(e)

    def fetch_all_data(self, document: str):
        try:
            doc_ref = self.db.collection(self.image_set).document(document)
            doc = doc_ref.get()
            if doc.exists:
                # doc.to_dict
                print('done ! ')
            else:
                print('No such document!')
            return doc.to_dict
        except Exception as e:
            print(e)

    def fetch_some_data(self, column: str, condition, value):
        try:
            doc_ref = self.db.collection(self.image_set)
            query_ref = doc_ref.where(column, condition, value).stream()
            results = []
            for doc in query_ref:
                results.append(doc.to_dict())
            print('done !')
            return results
        except Exception as e:
            print(e)

    def update_data(self, document, update_data: Dict):
        doc_ref = self.db.collection(self.image_set).document(document)
        doc_ref.update({update_data})

    def download_all_images(self, output_path: str, is_process: bool = False):
        try:
            out_path = Path(output_path)
            if out_path.exists():
                print('[WARNING] output folder is exist')
                return
            docs = self.db.collection(self.image_set)
            results = docs.where('is_processed', '==', is_process).stream()
            out_path.mkdir(parents=True, exist_ok=True)
            if results:
                for doc in results:
                    res = doc.to_dict()
                    sub_path = out_path / res['image_id']
                    sub_path.mkdir(parents=True, exist_ok=True)
                    for data in res['data']:
                        image = self.base64_to_image(data)
                        image.save(str(sub_path / (str(uuid.uuid4()) + '.jpg')))
                    del res['data']
                    with open(str(sub_path) + '/image_info.json', 'w', encoding='utf-8') as f:
                        json.dump(res, f)
                print('[INFO] Download done!')
            else:
                print('[INFO] fetch empty data')
        except Exception as e:
            print(e)
