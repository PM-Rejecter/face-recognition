import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Concatenate, \
    Layer
from tensorflow.keras.regularizers import l2

class NN1(Model):
    def __init__(self):
        super(NN1, self).__init__()

        # Build layers
        self.conv1 = Conv2D(64, 7, 2, name='conv1', activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name="pool1")
        self.rnorm1 = BatchNormalization(name="rnrom1")
        self.conv2a = Conv2D(64, 1, 1, name='conv1a', activation='relu', padding='same')
        self.conv2 = Conv2D(192, 3, 1, name='conv2', activation='relu', padding='same')
        self.rnorm2 = BatchNormalization(name="rnrom2")
        self.pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name="pool1")
        self.conv3a = Conv2D(192, 1, 1, name='conv3a', activation='relu', padding='same')
        self.conv3 = Conv2D(192, 3, 1, name='conv3', activation='relu', padding='same')
        self.pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name="pool1")
        self.conv4a = Conv2D(384, 1, 1, name='conv4a', activation='relu', padding='same')
        self.conv4 = Conv2D(384, 3, 1, name='conv4', activation='relu', padding='same')
        self.conv5a = Conv2D(256, 1, 1, name='conv5a', activation='relu', padding='same')
        self.conv5 = Conv2D(256, 3, 1, name='conv5', activation='relu', padding='same')
        self.conv6a = Conv2D(256, 1, 1, name='conv6a', activation='relu', padding='same')
        self.conv6 = Conv2D(256, 3, 1, name='conv6', activation='relu', padding='same')
        self.pool4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name="pool1")
        self.flatten = Flatten(name="flatten")
        self.reshape = Dense(4096, name="reshape")
        self.fc1 = tfa.layers.Maxout(32 * 128, name="fc1")
        self.fc2 = tfa.layers.Maxout(32 * 128, name="fc2")
        self.fc3 = Dense(128, activity_regularizer=l2(0.01), name="fc3")

    def call(self, inputs,training=True):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.rnorm1(x)
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.rnorm2(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4(x)
        x = self.conv5a(x)
        x = self.conv5(x)
        x = self.conv6a(x)
        x = self.conv6(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.reshape(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    nn1 = NN1()