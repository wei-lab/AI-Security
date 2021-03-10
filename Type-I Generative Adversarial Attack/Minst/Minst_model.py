import tensorflow as tf

class MyModels(tf.keras.Model):
    def __init__(self):
        super(MyModels, self).__init__()

        self.conv1=tf.keras.layers.Conv2D(10,kernel_size=3,strides=1,padding='same',name='conv1',activation=tf.nn.relu)
        self.pl1=tf.keras.layers.MaxPool2D(pool_size=2,strides=1)
        self.cbn1=tf.keras.layers.BatchNormalization()

        self.conv2=tf.keras.layers.Conv2D(32,kernel_size=3,strides=1,padding='same',name='conv2',activation=tf.nn.relu)
        self.pl2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=1)
        self.cbn2 = tf.keras.layers.BatchNormalization()

        self.conv3=tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same',name='conv3',activation=tf.nn.relu)
        self.cbn3 = tf.keras.layers.BatchNormalization()

        self.flatens=tf.keras.layers.Flatten()

        self.fc2=tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu)
        self.bn2=tf.keras.layers.BatchNormalization()
        self.fc3=tf.keras.layers.Dense(10,activation=tf.nn.softmax)

        self._set_inputs(inputs=tf.TensorSpec([None,28,28,1],dtype=tf.float32),training=None)

    def conve_layers(self,inputs):
        x=self.conv1(inputs)
        x=self.pl1(x)
        x=self.cbn1(x)
        x=self.conv2(x)
        x=self.pl2(x)
        x=self.cbn2(x)
        x=self.conv3(x)
        x=self.cbn3(x)
        x=self.flatens(x)
        return x

    def get_layer_feature(self,name,inputs):
        if name=='conv1':
            return self.conv1(inputs)
        elif name=='conv2':
            x=self.conv1(inputs)
            x=self.conv2(x)
            return x
        elif name=='conv3':
            x = self.conv1(inputs)
            x = self.conv2(x)
            x=self.conv3(x)
            return x
        else:
            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


    def Dense_layers(self,inputs,training=None):

        x=self.bn2(self.fc2(inputs),training)
        x=self.fc3(x)
        return x

    def call(self, inputs, training=None, mask=None):
        x=self.conve_layers(inputs)
        x=self.Dense_layers(x,training)

        return x


def main():
    t=tf.random.normal([128,28,28,1])
    print(t.shape)
    model=MyModels()
    x=model(t)
    print(x.shape)
    t2=model.get_layer_feature('conv1',t)
    print(t2.shape)


if __name__ == '__main__':
    main()
