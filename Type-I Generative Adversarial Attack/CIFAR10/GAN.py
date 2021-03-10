import tensorflow as tf
from tensorflow.keras import layers


class Genrater(tf.keras.Model):
    def __init__(self):
        super(Genrater, self).__init__()
        self.fc=layers.Dense(3*3*512)

        self.convtrans1=layers.Conv2DTranspose(512,5,2)
        self.bn1=layers.BatchNormalization()

        self.convtrans2= layers.Conv2DTranspose(256, 5, 1)
        self.bn2 = layers.BatchNormalization()

        self.convtrans3 = layers.Conv2DTranspose(128, 3, 2)
        self.bn3 = layers.BatchNormalization()

        self.convtrans4 = layers.Conv2DTranspose(64, 4, 1)
        self.bn4 = layers.BatchNormalization()

        self.convtrans5 = layers.Conv2DTranspose(3, 3, 1)

    def call(self,x,training=None):
        batch=x.shape[0]
        z=tf.random.uniform([batch,100],maxval=1.,minval=-1.0)
        z=self.fc(z)
        z=tf.reshape(z,[-1,3,3,512])
        z=self.bn1(tf.nn.leaky_relu(self.convtrans1(z)),training)
        z = self.bn2(tf.nn.leaky_relu(self.convtrans2(z)), training)
        z = self.bn3(tf.nn.leaky_relu(self.convtrans3(z)), training)
        z = self.bn4(tf.nn.leaky_relu(self.convtrans4(z)), training)
        z=self.convtrans5(z)
        z = tf.nn.tanh(z)
        z = tf.clip_by_value(z, -0.003, 0.003)
        x = tf.add(x, z)
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


