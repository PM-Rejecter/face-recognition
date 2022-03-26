import tensorflow
from DataBase.DataSummoner import DataSummoner
import glob
from pathlib import Path
from PIL import Image
import numpy as np
from face_net_nn1 import NN1
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

# ds = DataSummoner("exodia-face-recognition-firebase-adminsdk-1xywr-4f3dc249ba.json")
#
# ds.download_all_images("image/data", True)
# images_list = []
# images_label = []
# images_label_map = {}
# label_count = 0
# for path in glob.glob('image/data/*'):
#     images_label_map[label_count] = Path(path).name
#     for sub_path in glob.glob(path + '/*.jpg'):
#         image = Image.open(sub_path).resize((220,220))
#         image_array = np.array(image)
#         images_list.append(image_array)
#         images_label.append(label_count)
#     label_count += 1
# images_list = np.asarray(images_list,dtype=float)
# images_label = np.asarray(images_label,dtype=float)

train_ds = tf.keras.utils.image_dataset_from_directory(
    'lfw_funneled',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(220, 220),
    batch_size=32
)


def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

train_ds = train_ds.map(_normalize_img)

model = NN1()
loss_func = tfa.losses.TripletSemiHardLoss(margin=0.2)
optimizer = tfa.optimizers.RectifiedAdam(learning_rate=0.01)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

EPOCHS = 500

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    # val_loss.reset_states()
    # val_accuracy.reset_states()

    # for images, labels in zip(new_x_train,new_y_train):
    #     train_step(images, labels)
    for images, labels in train_ds:
        train_step(images, labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
    )

