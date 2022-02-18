import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2


def _reduce_conv_norm_pool(reduced_filters: int, conv_filters: int, norm: bool, pool: bool,
                           level: str = None):
    model = Sequential(name=f"level{level}")

    # add 1x1 reduced layer
    model.add(Conv2D(reduced_filters, 1, padding='same', name=f"conv{level}a"))
    # add conv2d
    model.add(Conv2D(conv_filters, 3, padding='same', name=f"conv{level}", activation='relu'))
    # add batch normalization
    if norm:
        model.add(BatchNormalization())
    # add max pooling
    if pool:
        model.add(MaxPooling2D((3, 3), 2, padding='same', name=f"pool{level}"))

    return model


class NN1(Model):
    def __init__(self):
        super(NN1, self).__init__()

        # Build layers
        self.conv1 = Conv2D(64, 7, 2, name='conv1', activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name="pool1")
        self.rnorm1 = BatchNormalization(name="rnrom1")

        self.block2 = _reduce_conv_norm_pool(64, 192, norm=True, pool=True, level=2)
        self.block3 = _reduce_conv_norm_pool(192, 384, norm=False, pool=True, level=3)
        self.block4 = _reduce_conv_norm_pool(384, 256, norm=False, pool=False, level=4)
        self.block5 = _reduce_conv_norm_pool(256, 256, norm=False, pool=False, level=5)
        self.block6 = _reduce_conv_norm_pool(256, 256, norm=False, pool=True, level=6)

        self.flatten = Flatten(name="flatten")
        self.reshape = Dense(4096, name="reshape")
        self.fc1 = tfa.layers.Maxout(32*128, name="fc1")
        self.fc2 = tfa.layers.Maxout(32*128, name="fc2")
        self.fc3 = Dense(128, activity_regularizer=l2(0.01), name="fc3")

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.rnorm1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.flatten(x)
        x = self.reshape(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class NN2(model):
    def __init__(self):
        super(NN2, self).__init__()


    def call(self):
        pass
