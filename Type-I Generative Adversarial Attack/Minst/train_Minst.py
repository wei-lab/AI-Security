import tensorflow as tf
from Minst_model import MyModels
from matplotlib import pyplot as plt
import numpy as np


tf.random.set_seed(100)
np.random.seed(100)
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
    model=MyModels()
    optimizer=tf.optimizers.Adam(0.0003)
    train_db,test_db=getData(batchs)
    max_acc=0.
    acc_list=[]
    epch_list=[]

    model.build(input_shape=(None,28,28,1))
    model.summary()

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
            model.save_weights('MnistModel/my_weights')
            max_acc=acc
        acc_list.append(acc)
        epch_list.append(epch)
        print("test data acc is:",float(acc))


    # model.save('ministmodel')
    # del model

def model_test():
    # re_model=tf.keras.models.load_model('ministmodel')
    _,test_db=getData(128)
    re_model=MyModels()
    re_model.load_weights('MnistModel/my_weights')
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
    model_train()
    # model_test()
