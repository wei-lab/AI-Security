
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import tensorflow as tf
import time

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


img_size = 32
img_chan = 3
n_classes = 10
_WEIGHT_DECAY = 0
BN_EPSILON = 0.001
weight_decay = 0.
batch_size = 128

padding_size = 2
print('\nLoading CIFAR10')

cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32)/127.5 - 1.0
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32)/127.5 - 1.0


to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train_mean = np.mean(X_train, axis=0)
X_train -= x_train_mean
X_test -= x_train_mean

datagen = ImageDataGenerator(
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # randomly flip images
    horizontal_flip=True)

datagen.fit(X_train)
data_iter = datagen.flow(X_train, y_train, batch_size=batch_size)


print('\nConstruction graph')

def isce_loss(embedding, labels, out_num, w_init=None, s=1., m=0.1):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels
    :param s: scalar value
    :param out_num: output class num
    :param m: add inference value, default is 0.1
    :return: the final output
    '''
    with tf.variable_scope('isce_loss'):
        # weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        t=embedding
#         mt=s*(t+m)
        mt=tf.clip_by_value(s*(t+m),-1,1)
        v = t 
        cond = tf.cast(tf.nn.relu(v, name='if_else'), dtype=tf.bool)
        keep_val = t
        mask = labels
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        output = tf.add(tf.multiply(t, inv_mask), tf.multiply(mt, mask), name='isce_loss_output')

    return output

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def model(input_tensor_batch,label=None,n=5, reuse=True,training = False):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    input_tensor_batch = tf.image.per_image_standardization(input_tensor_batch)
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        # conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1) #cifar10
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3,3, 16], 1)
        # activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 16)
            # activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32)
            # activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64)
            layers.append(conv3)
        
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])
        global_pool = tf.layers.dense(global_pool, units=64)
        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 10)
        layers.append(output)
        label = tf.argmax(output, 1)
        label = tf.one_hot(label, 10)
        output = isce_loss(output, labels=label, out_num=10)
        
        return output

def l1_norm_tf(input_x, epsilon=1e-24):
    """get L1 norm"""
    reduc_ind = list(range(1, len(input_x.get_shape())))
    return tf.reduce_sum(tf.abs(input_x),
                         reduction_indices=reduc_ind,
                         keep_dims=True) + epsilon


def l2_norm_tf(input_x, epsilon=1e-24):
    """get L2 norm"""
    reduc_ind = list(range(1, len(input_x.get_shape())))
    return tf.sqrt(tf.reduce_sum(tf.square(input_x),
                                 reduction_indices=reduc_ind,
                                 keep_dims=True)) + epsilon


def pgd(model, x, eps=0.1, epochs=100, label=None, sign=True, clip_min=-1.0, clip_max=1.):
    """
    Fast gradient method.
    """
    xadv = tf.identity(x)

    # ybar = np.array(model(xadv))[0]
    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        logits= model(xadv)
        loss = loss_fn(labels=target, logits=logits)

        dy_dx, = tf.gradients(loss, xadv)
        dy_dx = dy_dx / l2_norm_tf(dy_dx)
        xadv = tf.stop_gradient(xadv + eps * noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i + 1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
                            name='fast_gradient')
    return xadv


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.lr = tf.placeholder(tf.float32)
    env.training = tf.placeholder_with_default(False, (), name='mode')


    logits = model(env.x,env.y,n=5,reuse=False,training = env.training)
    env.ybar=tf.nn.softmax(logits)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss') + weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    # with tf.variable_scope('train_op'):
    #     optimizer = tf.train.AdamOptimizer(env.lr)
    #     env.train_op = optimizer.minimize(env.loss)
    with tf.variable_scope('train_op'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=env.lr, momentum=0.9)
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver(max_to_keep=1)

with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.pgd_eps = tf.placeholder(tf.float32, (), name='pgd_eps')
    env.pgd_epochs = tf.placeholder(tf.int32, (), name='pgd_epochs')
    env.x_fgmt = pgd(model, env.x, epochs=env.pgd_epochs, eps=env.pgd_eps)

print('\nInitializing graph')



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess, env, X_data, y_data, batch_size=256):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc

def gen(data_iter):
    X_data=[]
    y_data=[]
    for i in range(1000):
        x,y = data_iter.next()
        X_data.append(x)
        y_data.append(y)
    return np.array(X_data),np.array(y_data)

def train(sess, env, data_iter, X_valid=None, y_valid=None, steps=20000,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, './my_model/cifar10.ckpt')

    

    print('\nTrain model')
    
    os.makedirs('my_model', exist_ok=True)
    lr=0.1
    for step in range(steps//1000):

        # X_data,y_data = data_iter.next()
        X_data,y_data = gen(data_iter)
        # lr=0.95*lr
        if step==10:
            lr=0.01
        if step==15:
            lr=0.001
        # for X_data,y_data in data_iter.next():
        for i in range(1000):
            sess.run(env.train_op, feed_dict={env.x: X_data[i],env.y: y_data[i], env.lr: lr, env.training: True})

                                   
            if i%500==0:
                print('i',i)
                if X_valid is not None:
                    evaluate(sess, env, X_valid, y_valid)
        print('step',step)

        env.saver.save(sess,'./my_model/cifar10.ckpt')



def make_fgmt(sess, env, X_data, epochs=100, eps=0.1, batch_size=256):
    """
    Generate pgd by running env.x_pgd.
    """
    print('\nMaking adversarials via pgd')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgmt, feed_dict={
            env.x: X_data[start:end],
            env.pgd_eps: eps,
            env.pgd_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv



print('\nTraining')

train(sess, env, data_iter, X_test, y_test, load=False, name='cifar10')

evaluate(sess,env,X_test,y_test)


epochs = [5, 50]
for i in epochs:
    X_adv = make_fgmt(sess, env, X_test, eps=0.01, epochs=i)
    print('\nEvaluating on adversarial data epochs=%d' % i)
    evaluate(sess, env, X_adv, y_test)
epochs = [5, 50]
for i in epochs:
    X_adv = make_fgmt(sess, env, X_test, eps=0.04, epochs=i)
    print('\nEvaluating on adversarial data epochs=%d' % i)
    evaluate(sess, env, X_adv, y_test)