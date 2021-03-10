import tensorflow as tf
from CIFARModel import MyModel

def process(x,y):
    x=2*tf.cast(x,dtype=tf.float32)/255.-1
    y=tf.cast(y,dtype=tf.int32)
    return x,y

def getData(batchs):
    (train_x,train_y),(test_x,test_y)=tf.keras.datasets.cifar10.load_data()

    train_y=tf.squeeze(train_y)
    test_y=tf.squeeze(test_y)

    train_db=tf.data.Dataset.from_tensor_slices((train_x,train_y))
    train_db=train_db.map(process).shuffle(1000).batch(batchs)

    test_db=tf.data.Dataset.from_tensor_slices((test_x,test_y))
    test_db=test_db.map(process).batch(batchs)

    return train_db,test_db

def model_train():
    epochs=500
    model=MyModel()
    max_acc=0.
    optimizer=tf.optimizers.Adam(lr=0.0003)

    train_db,test_db=getData(128)

    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape()  as tape:
                logits=model(x,training=True)
                y_hot=tf.one_hot(y,depth=10)
                loss=tf.losses.categorical_crossentropy(y_hot,logits,from_logits=True)
                loss=tf.reduce_mean(loss)

            grads=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            if step % 100 ==0:
                print("{} train data loss is: {}".format(epoch,float(loss)))

        total_current=0
        total_number=0
        for x,y in test_db:
            prob=model(x,training=False)
            prob=tf.argmax(prob,axis=1)
            prob=tf.cast(prob,dtype=tf.int32)
            current=tf.equal(prob,y)
            current=tf.cast(current,dtype=tf.int32)
            current=tf.reduce_sum(current)
            total_current+=current
            total_number+=x.shape[0]

        acc=total_current/total_number
        if acc>=max_acc:
            model.save_weights("Save_Model/my_weights.kpl")
            max_acc=acc
        print("test acc is: ",float(acc))


def model_test():
    model=MyModel()
    _,test_db=getData(128)
    model.load_weights("Save_Model/my_weights.kpl")
    total_current = 0
    total_number = 0
    for x, y in test_db:
        prob = model(x, training=False)
        prob = tf.argmax(prob, axis=1)
        prob = tf.cast(prob, dtype=tf.int32)
        current = tf.equal(prob, y)
        current = tf.cast(current, dtype=tf.int32)
        current = tf.reduce_sum(current)
        total_current += current
        total_number += x.shape[0]

    acc = total_current / total_number
    print("remodel test acc is: ", float(acc))
    t=tf.random.normal([128,32,32,3])
    tt=model.get_layer_feature('conv3',t)
    print(tt.shape)

if __name__ == '__main__':
    # model_train()
    model_test()
