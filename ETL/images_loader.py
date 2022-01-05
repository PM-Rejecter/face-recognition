import glob
from DataBase.DataSummoner import DataSummoner
from pathlib import Path
import uuid
ds = DataSummoner("../DataBase/exodia-face-recognition-firebase-adminsdk-1xywr-4f1f07fee9.json")


for path in glob.glob('../test_image/*'):
    insert_data = {}
    p = Path(path)
    data = []
    for sub_path in glob.glob(path+'/*.jpg'):
        img_base64 = ds.image_to_base64(sub_path)
        data.append(img_base64)
    file = p.name
    ds.save_data(data={'data':data,'name':file},id=str(file))