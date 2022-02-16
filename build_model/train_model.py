from pathlib import Path

import tensorflow as tf

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

# import model
model = NN1()

# setting optimizer and loss function

