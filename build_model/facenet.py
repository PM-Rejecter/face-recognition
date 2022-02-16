import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2


def _reduce_conv_norm_pool(reduced_filters: int, conv_filters: int, norm: bool, pool: bool,
                           level: str = None):
    model = Sequential(name=f"level{level}")

    # add 1x1 reduced layer
    model.add(Conv2D(reduced_filters, 1, name=f"conv{level}a"))
    # add conv2d
    model.add(Conv2D(conv_filters, 3, name=f"conv{level}", activation='relu'))
    # add batch normalization
    if norm:
        model.add(BatchNormalization())
    # add max pooling
    if pool:
        model.add(MaxPooling2D((3, 3), 2, name=f"pool{level}"))

    return model


class NN1(Model):
    def __init__(self):
        super(NN1, self).__init__()

        # Build layers
        self.conv1 = Conv2D(64, 7, 2, name='conv1', activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, name="pool1")
        self.rnorm1 = BatchNormalization(name="rnrom1")

        self.block2 = _reduce_conv_norm_pool(64, 192, norm=True, pool=True, level=2)
        self.block3 = _reduce_conv_norm_pool(192, 384, norm=False, pool=True, level=3)
        self.block4 = _reduce_conv_norm_pool(384, 256, norm=False, pool=False, level=4)
        self.block5 = _reduce_conv_norm_pool(256, 256, norm=False, pool=False, level=5)
        self.block6 = _reduce_conv_norm_pool(256, 256, norm=False, pool=True, level=6)

        self.flatten = Flatten()
        self.fc1 = tfa.layers.Maxout(32 * 128)
        self.fc2 = tfa.layers.Maxout(32 * 128)
        self.fc3 = Dense(128, activity_regularizer=l2(0.01))

    def call(self, inputs, training=None, mask=None):
        model = Sequential([
            self.conv1,
            self.pool1,
            self.rnorm1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.block6,
            self.flatten,
            self.fc1,
            self.fc2,
            self.fc3
        ])
        return model(inputs)
