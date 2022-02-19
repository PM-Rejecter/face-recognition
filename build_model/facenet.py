import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Concatenate, \
    Layer, AveragePooling2D
from tensorflow.keras.regularizers import l2

import tensorflow as tf


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


def l2_norm2d(x, pool_size=(2, 2), strides=None,
              padding='valid', data_format=None):
    if strides is None:
        strides = pool_size
    x = x ** 2
    output = K.pool2d(x, pool_size, strides,
                      padding, data_format, pool_mode='avg')
    output = K.sqrt(output)
    return output


class Inception(Layer):
    def __init__(self, c1_filters: int, c3_reduced_filters: int, c3_filters: int, c5_reduced_filters: int,
                 c5_filters: int, pool_proj: str, cp_filters: int = None, c3_strides: int = None,
                 c5_strides: int = None):
        super(Inception, self).__init__()
        self.pool_proj = pool_proj
        self.cp_filters = cp_filters
        self.c1_filters = c1_filters

        # Build layers
        self.p1_1 = Conv2D(c1_filters, 1)

        self.p2_1 = Conv2D(c3_reduced_filters, 1)
        if c3_strides:
            self.p2_2 = Conv2D(c3_filters, 3, strides=(c3_strides, c3_strides), padding='same')
        else:
            self.p2_2 = Conv2D(c3_filters, 3, padding='same')

        self.p3_1 = Conv2D(c5_reduced_filters, 1)
        if c5_strides:
            self.p3_2 = Conv2D(c5_filters, 5, strides=(c5_strides, c5_strides), padding='same')
        else:
            self.p3_2 = Conv2D(c5_filters, 5, padding='same')

        if cp_filters:
            self.p4_max = MaxPooling2D((3, 3), strides=1, padding='same')
            self.p4_conv = Conv2D(cp_filters, 1)
        else:
            self.p4_max = MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        self.p4_l2 = Lambda(lambda x: l2_norm2d(x, (3, 3), strides=(1, 1), padding='same'))

    def call(self, inputs, training=None, **kwargs):
        pool_proj, cp_filters, c1_filters = self.pool_proj, self.cp_filters, self.c1_filters

        # 1x1 Conv2D
        if c1_filters:
            partition_1 = self.p1_1(inputs)  # check if it works while c1_filters = 0

        # 3x3 Conv2D
        partition_2 = self.p2_1(inputs)
        partition_2 = self.p2_2(partition_2)

        # 5x5 Conv2D
        partition_3 = self.p3_1(inputs)
        partition_3 = self.p3_2(partition_3)

        # pooling
        if pool_proj == 'maxpooling':
            partition_4 = self.p4_max(inputs)
        elif pool_proj == 'l2':
            partition_4 = self.p4_l2(inputs)
        else:
            raise Exception(f'pool_proj should be maxpooling | l2, received {pool_proj} instead.')

        if cp_filters:
            partition_4 = self.p4_conv(partition_4)

        if c1_filters:
            output = Concatenate(axis=-1)([partition_1, partition_2, partition_3, partition_4])
        else:
            output = Concatenate(axis=-1)([partition_2, partition_3, partition_4])
        return output


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
        self.fc1 = tfa.layers.Maxout(32 * 128, name="fc1")
        self.fc2 = tfa.layers.Maxout(32 * 128, name="fc2")
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


class NN2(Model):
    def __init__(self):
        super(NN2, self).__init__()

        # Build layers
        self.conv1 = Conv2D(64, 7, 2, padding='same', name='conv1')
        self.maxpooling = MaxPooling2D((3, 3), 2, padding='same')
        self.norm1 = BatchNormalization()

        self.inception2_1 = Conv2D(64, 1)
        self.inception2_2 = Conv2D(192, 3, padding='same')
        self.norm2 = BatchNormalization()

        self.inception3a = Inception(64, 96, 128, 16, 32, 'maxpooling', 32)
        self.inception3b = Inception(64, 96, 128, 32, 64, 'l2', 64)
        self.inception3c = Inception(0, 128, 256, 32, 64, 'maxpooling', c3_strides=2, c5_strides=2)

        self.inception4a = Inception(256, 96, 192, 32, 64, 'l2', 128)
        self.inception4b = Inception(224, 112, 224, 32, 64, 'l2', 128)
        self.inception4c = Inception(192, 128, 256, 32, 64, 'l2', 128)
        self.inception4d = Inception(160, 144, 288, 32, 64, 'l2', 128)
        self.inception4e = Inception(0, 160, 256, 64, 128, 'maxpooling', c3_strides=2, c5_strides=2)

        self.inception5a = Inception(384, 192, 384, 48, 128, 'l2', 128)
        self.inception5b = Inception(384, 192, 384, 48, 128, 'maxpooling', 128)

        self.avgpooling = AveragePooling2D((7, 7))
        self.flatten = Flatten()
        self.fully = Dense(128, 'relu')
        self.l2 = Lambda(lambda x: K.l2_normalize(x, axis=0))

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.maxpooling(x)
        x = self.norm1(x)

        x = self.inception2_1(x)
        x = self.inception2_2(x)

        x = self.norm2(x)
        x = self.maxpooling(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpooling(x)
        x = self.flatten(x)
        x = self.fully(x)
        x = self.l2(x, -1)
        return x
