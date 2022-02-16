from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa

from build_model.facenet import NN1

# 限制 gpu 的記憶體使用量，不要一次抓死
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 讀入資料集
data_dir = Path('data')

# 資料參數
batch_size = 32
img_height = 220
img_width = 220

# train/val datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# rescale image
def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return img, label


train_ds = train_ds.map(_normalize_img)
val_ds = val_ds.map(_normalize_img)

# import model
model = NN1()

# setting optimizer and loss function
loss = tfa.losses.TripletSemiHardLoss(margin=0.2)
optimizer = tf.keras.optimizers.Adam()

