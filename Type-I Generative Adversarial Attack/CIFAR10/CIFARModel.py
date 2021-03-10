import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1=layers.Conv2D(64,3,1,'same',name='conv1')
        self.pool1=layers.MaxPool2D(2,2)
        self.bn1=layers.BatchNormalization()

        self.conv2 = layers.Conv2D(128, 3, 1, 'same',name='conv1')
        self.pool2= layers.MaxPool2D(2, 2)
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, 3, 1, 'same',name='conv1')
        self.pool3 = layers.MaxPool2D(2, 2)
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(512, 3, 1, 'same',name='conv1')
        self.pool4 = layers.MaxPool2D(2, 2)
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(512, 3, 1, 'same',name='conv1')
        self.pool5 = layers.MaxPool2D(2, 2)
        self.bn5 = layers.BatchNormalization()

        self.flatten=layers.Flatten()
        self.fc1=layers.Dense(256)
        self.fc2=layers.Dense(10)

        self._set_inputs(inputs=tf.TensorSpec([None,32,32,3],dtype=tf.float32),training=None)

    def get_layer_feature(self,name,x,training=False):
        if name == 'conv1':
            return self.conv1(x)
        elif name == 'conv2':
            x = self.conv1(x)
            x=self.pool1(x)
            return self.conv2(x)
        elif name == 'conv3':
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            return self.conv3(x)
        elif name == 'conv4':
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            return self.conv4(x)
        else:
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            x = self.conv4(x)
            x = self.pool4(x)
            return self.conv5(x)



    def call(self,x,training=None):
        x=self.bn1(tf.nn.relu(self.pool1(self.conv1(x))),training)
        x = self.bn2(tf.nn.relu(self.pool2(self.conv2(x))), training)
        x = self.bn3(tf.nn.relu(self.pool3(self.conv3(x))), training)
        x = self.bn4(tf.nn.relu(self.pool4(self.conv4(x))), training)
        x = self.bn5(tf.nn.relu(self.pool5(self.conv5(x))), training)
        x=self.flatten(x)
        x=self.fc1(x)
        x=tf.nn.relu(x)
        x=self.fc2(x)
        return x


if __name__ == '__main__':
    model=MyModel()
    x=tf.random.normal([128,32,32,3])
    tt=model(x)
    print(tt.shape)
    t=model.get_layer_feature('conv3',x)
    print(t.shape)

