import cv2
import numpy as np
from PIL import Image
from embedding_builder import EmbeddingBuilder
from preprocessing.image_preprocess import get_processed_image
import tensorflow as tf

eb = EmbeddingBuilder(model_path='face_model')
test_image = get_processed_image('test2.jpg')
eb.new_embedding(test_image, 'hawk')
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')

    # Resizing into dimensions you used while training
    im = im.resize((224, 224))
    img_array = np.array(im)
    pred_img = get_processed_image(img_array, False)
    if len(pred_img) == 0:
        continue
    pred_img = cv2.resize(pred_img, (224, 224))
    pred_img = np.reshape(pred_img, (1, pred_img.shape[0], pred_img.shape[1], pred_img.shape[2]))
    pred_img = pred_img / 255
    pred_img = tf.convert_to_tensor(pred_img, dtype=tf.float32)
    # Expand dimensions to match the 4D Tensor shape.
    # img_array = np.expand_dims(pred_img, axis=0)
    eb.predict(pred_img)

    cv2.imshow("Prediction", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
