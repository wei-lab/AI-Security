import tensorflow as tf
from tensorflow.keras import layers
class Dens_Model(tf.keras.Model):
    def __init__(self):
        super(Dens_Model, self).__init__()
        self.fc=tf.keras.Sequential([
            layers.Dense(784),
            layers.ReLU(),

            layers.Dense(512),
            layers.ReLU(),

            layers.Dense(256),
            layers.ReLU(),

            layers.Dense(128),
            layers.ReLU(),

            layers.Dense(10)

        ])


    def call(self,x):
        x=self.fc(x)
        return x

def process(x,y):
    x=2*tf.cast(x,dtype=tf.float32)/255.-1
    y=tf.cast(y,dtype=tf.int32)
    return x,y

def getData(batchs):
    (train_x,train_y),(test_x,test_y)=tf.keras.datasets.mnist.load_data()

    train_y=tf.squeeze(train_y)
    test_y=tf.squeeze(test_y)

    train_db=tf.data.Dataset.from_tensor_slices((train_x,train_y))
    train_db=train_db.map(process).shuffle(1000).batch(batchs)

    test_db=tf.data.Dataset.from_tensor_slices((test_x,test_y))
    test_db=test_db.map(process).batch(batchs)

    return train_db,test_db

def model_train():
    epochs=200
    batch_size=128
    optimizer=tf.optimizers.Adam(lr=0.001)
    model=Dens_Model()
    max_acc=0

    train_db,test_db=getData(batch_size)

    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                x=tf.reshape(x,[-1,784])
                logits=model(x)
                y_hot=tf.one_hot(y,depth=10)
                loss=tf.losses.categorical_crossentropy(y_hot,logits,from_logits=True)
                loss=tf.reduce_mean(loss)

            grad=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grad,model.trainable_variables))
            if step % 100 ==0:
                print("{} epoch train loss is :{}".format(epoch,float(loss)))

        total_number=0
        total_current=0
        for x,y in test_db:
            x=tf.reshape(x,[-1,784])
            prob=model(x)
            prob=tf.argmax(prob,axis=1)
            prob=tf.cast(prob,dtype=tf.int32)
            current=tf.equal(prob,y)
            current=tf.cast(current,dtype=tf.int32)
            current=tf.reduce_sum(current)
            total_current+=current
            total_number+=x.shape[0]

        acc=total_current/total_number
        if acc>=max_acc:
            model.save_weights("MnistModel/dens_weights.kpl")
            max_acc=acc
        print("acc is : ",float(acc))

def model_test():
    _,test_db=getData(128)
    model=Dens_Model()
    model.load_weights("MnistModel/dens_weights.kpl")

    total_number = 0
    total_current = 0
    for x, y in test_db:
        x = tf.reshape(x, [-1, 784])
        prob = model(x)
        prob = tf.argmax(prob, axis=1)
        prob = tf.cast(prob, dtype=tf.int32)
        current = tf.equal(prob, y)
        current = tf.cast(current, dtype=tf.int32)
        current = tf.reduce_sum(current)
        total_current += current
        total_number += x.shape[0]

    acc = total_current / total_number
    print("test model acc is : ", float(acc))

if __name__ == '__main__':
    # x=tf.random.normal([128,784])
    # model=Dens_Model()
    # x=model(x)
    # print(x.shape)
    # model_train()
    model_test()