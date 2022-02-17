import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path

from build_model.facenet import NN1

# 限制 gpu 的記憶體使用量，不要一次抓死
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 讀入資料集
data_dir = Path('../data')

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
loss_object = tfa.losses.TripletSemiHardLoss(margin=0.2)
optimizer = tf.keras.optimizers.Adam()

# setting evaluate metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')


# train data function
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    val_loss(t_loss)
    val_accuracy(labels, predictions)


# training
EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in val_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Valuation Loss: {val_loss.result()}, '
        f'Valuation Accuracy: {val_accuracy.result() * 100}'
    )