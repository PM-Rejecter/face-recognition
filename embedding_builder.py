from keras import models
from collections import defaultdict
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_addons as tfa


class EmbeddingBuilder:
    def __init__(self, model_path: str):
        self.model = models.load_model(model_path)
        self.embedding_map = defaultdict(list)

    def new_embedding(self, image, image_name):
        img = cv2.resize(image, (224, 224))
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = img / 255
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        embedding = self.model.predict(img)
        self.embedding_map[image_name].append(embedding)

    def _normalize_img(img, label):
        img = tf.cast(img, tf.float32) / 255.
        return (img, label)

    def predict(self, image):
        # img = cv2.resize(image, (224, 224))
        img = image / 255
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        current_image_emb = self.model.predict(img)
        for name, embeddings in self.embedding_map.items():
            for embedding in embeddings:
                dist = np.linalg.norm(current_image_emb - embedding, ord=2)
                if dist <= 3730:
                    print("is {0} , {1}".format(name,dist))
                else:
                    print("unknown", dist)
