import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

tf.random.set_seed(100)

np.random.seed(100)


class Generater(tf.keras.Model):

    def __init__(self):
        super(Generater, self).__init__()

        self.conv = tf.keras.Sequential([
            layers.Conv2D(64, 3, 1, 'same'),
            layers.MaxPool2D(2, 2),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2D(120, 3, 1),
            layers.MaxPool2D(2, 2),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2D(512, 3, 1),
            layers.MaxPool2D(2, 2),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Flatten()
        ])

        self.fc = tf.keras.layers.Dense(3 * 3 * 512, activation=tf.nn.leaky_relu)
        self.contrans1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=1, strides=1,
                                                         activation=tf.nn.leaky_relu)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.contrans2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2,
                                                         activation=tf.nn.leaky_relu)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.contrans3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2,
                                                         activation=tf.nn.leaky_relu)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.contrans4 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=2, strides=2)

        # self._set_inputs(inputs=tf.TensorSpec([None, 100], dtype=tf.float32), training=None)

    def convtrains_layers(self, input_x, training):
        x = self.bn1(self.contrans1(input_x), training)
        x = self.bn2(self.contrans2(x), training)
        x = self.bn3(self.contrans3(x), training)
        x = self.contrans4(x)

        return x

    def call(self, tag, original, training=None):
        x = self.conv(original)
        batch_size = original.shape[0]
        batch_z = tf.random.uniform([batch_size, 100], minval=-1., maxval=1.)
        x = tf.concat([x, batch_z], axis=1)
        x = self.fc(x)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = self.convtrains_layers(x, training)
        x = tf.clip_by_value(x, -0.03, 0.03)
        x = tf.add(tag, x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv2D(32, 3, 1, 'same'),
            layers.MaxPool2D(2, 2),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2D(64, 3, 1, 'same'),
            layers.MaxPool2D(2, 2),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2D(128, 3, 1, 'same'),
            layers.MaxPool2D(2, 2),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2D(256, 3, 1, 'same'),
            layers.MaxPool2D(2, 2),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2D(512, 3, 1, 'same'),
            # layers.MaxPool2D(2,2),
            layers.BatchNormalization(),
            layers.LeakyReLU()
        ])

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(128)
        self.cress = layers.Dense(10)
        self.vality = layers.Dense(1)

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        label = self.cress(x)
        vality = self.vality(x)

        return vality, label
