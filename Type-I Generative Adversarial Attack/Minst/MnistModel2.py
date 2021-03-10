import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class M_Model(tf.keras.Model):
    def __init__(self):
        super(M_Model, self).__init__()
        self.conv1=layers.Conv2D(64,3,1,'same')
        self.pool1=layers.MaxPool2D(2,2)
        self.bn1=layers.BatchNormalization()

        self.conv2 = layers.Conv2D(128, 3, 1, 'same')
        self.pool2 = layers.MaxPool2D(2, 2)
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(512, 3, 2, 'same')
        self.pool3 = layers.MaxPool2D(2, 2)
        self.bn3 = layers.BatchNormalization()

        self.faltten=layers.Flatten()

        self.fc1=layers.Dense(256)
        self.fc2=layers.Dense(10)

    def call(self,x,training=None):
        x=self.bn1(self.pool1(self.conv1(x)),training)
        x=tf.nn.relu(x)
        x = self.bn2(self.pool2(self.conv2(x)), training)
        x = tf.nn.relu(x)
        x = self.bn3(self.pool3(self.conv3(x)), training)
        x = tf.nn.relu(x)
        # print(x.shape)
        x=self.faltten(x)
        x=self.fc1(x)
        x=self.fc2(x)
        return x

def process(x,y):
    x=2*tf.cast(x,dtype=tf.float32)/255-1
    y=tf.cast(y,dtype=tf.int32)
    return x,y

def getData(batchs):
    (train_x,train_y),(test_x,test_y)=tf.keras.datasets.fashion_mnist.load_data()
    # print(train_x.shape,max(train_y))
    train_x=tf.expand_dims(train_x,axis=3)
    test_x=tf.expand_dims(test_x,axis=3)

    train_db=tf.data.Dataset.from_tensor_slices((train_x,train_y))
    train_db=train_db.map(process).shuffle(1000).batch(batchs)

    test_db=tf.data.Dataset.from_tensor_slices((test_x,test_y))
    test_db=test_db.map(process).batch(batchs)

    return train_db,test_db

def model_train():
    batchs=128
    model=M_Model()
    optimizer=tf.optimizers.Adam(0.0001)
    train_db,test_db=getData(batchs)
    max_acc=0.


    for epch in range(250):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits=model(x)
                y_hot=tf.one_hot(y,depth=10)
                loss=tf.losses.categorical_crossentropy(y_hot,logits,from_logits=True)
                loss=tf.reduce_mean(loss)

            grads=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))

            if step% 100==0:
                print("{} epch train data loss is:{}".format(epch,float(loss)))

        total_current=0
        total_number=0
        for x,y in test_db:
            r=model(x)
            prob=tf.argmax(r,axis=1)
            prob=tf.cast(prob,dtype=tf.int32)
            y=tf.squeeze(y)
            prediction=tf.equal(y,prob)
            prediction=tf.cast(prediction,dtype=tf.int32)
            current=tf.reduce_sum(prediction)
            total_current+=current
            total_number+=x.shape[0]

        acc=total_current/total_number
        if acc>=max_acc:
            model.save_weights('MnistModel/my_weights2.kpl')
            max_acc=acc
        print("test data acc is:",float(acc))


    # model.save('ministmodel')
    # del model

def model_test():
    # re_model=tf.keras.models.load_model('ministmodel')
    _,test_db=getData(128)
    re_model=M_Model()
    re_model.load_weights('MnistModel/my_weights2.kpl')
    total_current = 0
    total_number = 0
    for x, y in test_db:
        r = re_model(x)
        prob = tf.argmax(r, axis=1)
        prob = tf.cast(prob, dtype=tf.int32)
        y = tf.squeeze(y)
        prediction = tf.equal(y, prob)
        prediction = tf.cast(prediction, dtype=tf.int32)
        current = tf.reduce_sum(prediction)
        total_current += current
        total_number += x.shape[0]

    acc = total_current / total_number
    print('re_model test data acc is:',float(acc))

if __name__ == '__main__':
    # x=tf.random.normal([128,28,28,1])
    # print(x.shape)
    # model=M_Model()
    # x=model(x)
    # print(x.shape)
    model_train()
    # model_test()
