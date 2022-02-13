import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class NN1(Model):
    def __init__(self):
        super(NN1, self).__init__()
        # self.conv1 = Conv2D()