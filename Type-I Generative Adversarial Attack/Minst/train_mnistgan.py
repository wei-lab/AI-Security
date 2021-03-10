import os
import numpy as np
import tensorflow as tf
from PIL import Image
from MinstGan import Generater,Discriminator
from Minst_model import MyModels
from MnistModel2 import M_Model
from MnistModel3 import Dens_Model

tf.random.set_seed(100)
np.random.seed(100)

def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)

def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))

    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))

    return tf.reduce_mean(loss)

def celoss_label(logits,label):
    loss=tf.losses.categorical_crossentropy(logits,label,from_logits=True)
    return tf.reduce_mean(loss)

def model_layer_feature(model,r_image,f_image):
    names=['conv1','conv2','conv3']
    diffrent=0.
    # logits_r=model.get_layer_feature('conv3',r_image)
    # logits_r*=-1
    # logits_f=model.get_layer_feature('conv3',f_image)
    # diffrent=tf.losses.mse(logits_r,logits_f)
    # diffrent=tf.reduce_mean(diffrent)
    for name in names:
        logits_r=model.get_layer_feature(name,r_image)
        logits_r=logits_r*-1
        logits_f=model.get_layer_feature(name,f_image)

        diff=tf.losses.mse(logits_r,logits_f)
        diff=tf.reduce_mean(diff)
        diffrent+=diff
    diffrent/=3
    return diffrent




def d_loss_fn(generator, discriminator, batch_original, batch_tag,orig_label,tag_label,f_model, is_training):
    fake_image = generator(batch_tag,batch_original, is_training)
    d_fake_vality,d_fake_prob = discriminator(fake_image, is_training)
    d_real_vality,d_real_prob = discriminator(batch_tag, is_training)

    d_loss_real = celoss_ones(d_real_vality)+celoss_label(d_real_prob,tag_label)
    d_loss_fake = celoss_zeros(d_fake_vality)+celoss_label(d_fake_prob,orig_label)

    f_model_loss=model_layer_feature(f_model,batch_original,fake_image)

    loss = (d_loss_fake + d_loss_real)+f_model_loss
    return loss,f_model_loss

def fmodel_loss(model,inputs,y):
    logits=model(inputs)
    y_hot=tf.one_hot(y,depth=10)
    loss=tf.losses.categorical_crossentropy(y_hot,logits,from_logits=True)
    loss=tf.reduce_mean(loss)
    return loss

def g_loss_fn(generator, discriminator,batch_tag, batch_orig,orig_label,is_training):
    fake_image = generator(batch_tag,batch_orig, is_training)
    d_fake_vality,d_fake_prob = discriminator(fake_image,orig_label, is_training)
    loss = celoss_ones(d_fake_vality)+celoss_label(d_fake_prob,orig_label)
    loss=loss
    return loss


def process(x,y):
    x=2*tf.cast(x,dtype=tf.float32)/255.-1
    y=tf.cast(y,dtype=tf.int32)

    return x,y


def show_rult(model,inputs,y):
    logits=model(inputs)
    prob=tf.argmax(logits,axis=1)
    prob = tf.cast(prob, dtype=tf.int32)
    y = tf.squeeze(y)
    prediction = tf.equal(y, prob)
    prediction = tf.cast(prediction, dtype=tf.int32)
    current = tf.reduce_sum(prediction)
    acc=current/inputs.shape[0]
    print("test attack is :",1.0-float(acc))

def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    z_dim = 100
    epochs = 100000
    batch_size = 128
    learning_rate = 0.0001
    is_training = True


    tag_data,orig_data=getData(batch_size)
    tag_data=tag_data.repeat()
    tag_iter=iter(tag_data)
    orig_data=orig_data.repeat()
    orig_iter=iter(orig_data)


    generator = Generater()
    discriminator = Discriminator()


    re_model=MyModels()
    re_model.load_weights('MnistModel/my_weights')

    re_model2 = M_Model()
    re_model2.load_weights('MnistModel/my_weights2.kpl')

    re_model3 = Dens_Model()
    re_model3.load_weights("MnistModel/dens_weights.kpl")

    g_optimizer = tf.optimizers.Adam(lr=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(lr=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        # batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_tag,tag_label= next(tag_iter)
        batch_orig,orig_label=next(orig_iter)

        for i in range(3):
            with tf.GradientTape() as tape:
                d_loss,f_loss = d_loss_fn(generator, discriminator, batch_orig, batch_tag,orig_label,tag_label,re_model, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator,batch_tag, batch_orig,orig_label, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

if __name__ == '__main__':
    main()
