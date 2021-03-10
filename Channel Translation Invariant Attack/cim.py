import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
from PIL import Image
import sys
import time
from nets import inception_v3, inception_v4,inception_resnet_v2,resnet_v2
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
slim = tf.contrib.slim
from tensorflow import reduce_sum
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# 声明一些攻击参数
CHECKPOINTS_DIR = './models'
model_checkpoint_map = {
    'inception_resnet_v2': os.path.join(CHECKPOINTS_DIR,'inception_resnet_v2.ckpt'),
    'ens_inception_resnet_v2': os.path.join(CHECKPOINTS_DIR,'ens_adv_inception_resnet_v2_rename'),
    'ens3_inception_v3': os.path.join(CHECKPOINTS_DIR,'ens3_adv_inception_v3_rename'),
    'inception_v4': os.path.join(CHECKPOINTS_DIR, 'inception_v4.ckpt'),
    'inception_v3': os.path.join(CHECKPOINTS_DIR, 'inception_v3.ckpt'),
    'resnet_v2_101': os.path.join(CHECKPOINTS_DIR, 'resnet_v2_101.ckpt'),
    'ens4_inception_v3': os.path.join(CHECKPOINTS_DIR, 'ens4_adv_inception_v3_rename')}

parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--input_dir', type=str, default='./ImageNet/images')
parser.add_argument('--output_dir', type=str, default='./data/test/')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_epsilon', type=int, default=16)
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--momentum', type=float, default=1.0)
parser.add_argument('--DIM', type=bool, default=False)
parser.add_argument('--prob', type=float, default=0.5)
parser.add_argument('--image_width', type=float, default=299)
parser.add_argument('--image_height', type=float, default=299)
parser.add_argument('--image_resize', type=float, default=330)
parser.add_argument('--cov', type=float, default=0.3)
FLAGS = parser.parse_args()

def gkern(kernlen=21, nsig=3, cov=0.1):
    x = np.linspace(-nsig, nsig, kernlen)
    # kern1d = st.norm.pdf(x)
    kern1d = st.multivariate_normal.pdf(x, mean=0, cov=cov)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

kernel = gkern(15, 3, FLAGS.cov).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.stack([stack_kernel, stack_kernel, stack_kernel]).transpose(1,2,3,0)

# load images
def load_images_with_true_label(input_dir):
    ori_images = []
    images = []
    filenames = []
    true_labels = []
    idx = 0
    label = pd.read_csv(os.path.join('./ImageNet', 'labels.csv'))
    filename2label = {label.iloc[i]['ImageId'] : label.iloc[i]['TrueLabel'] for i in range(len(label))}
    for filename in filename2label.keys():
        ori_image = Image.open(os.path.join(input_dir, filename), mode='r')
        ori_image = np.asarray(ori_image,dtype='float32')
        ori_images.append(ori_image)
        image = 2.0*(ori_image/255.0)-1.0
        images.append(image)
        filenames.append(filename)
        true_labels.append(filename2label[filename])
        idx += 1
        if idx == FLAGS.batch_size:
            images = np.array(images)
            ori_images = np.array(ori_images)
            yield ori_images, filenames, images, true_labels
            ori_images = []
            filenames = []
            images = []
            true_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        ori_images = np.array(ori_images)
        yield ori_images, filenames, images, true_labels

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')


def non_target_graph(x, y, i, x_max, x_min, grad): 
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  num_iter = FLAGS.num_iter
  alpha = eps / num_iter
  momentum = FLAGS.momentum
  num_classes = 1001

  image = x

  if FLAGS.DIM:
    image = input_diversity(x)
  
  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_inc_v3, end_points_inc_v3 = inception_v3.inception_v3(
        image, num_classes=num_classes, is_training=False, scope='InceptionV3')

  # with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
  #   logits_inc_v4, end_points_inc_v4 = inception_v4.inception_v4(
  #       image, num_classes=num_classes, is_training=False, scope='InceptionV4')

  # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
  #   logits_inc_res_v2, end_points_inc_res_v2 = inception_resnet_v2.inception_resnet_v2(
  #       image, num_classes=num_classes, is_training=False, scope='InceptionResnetV2')

  # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
  #   logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
  #       image, num_classes=num_classes, is_training=False, scope='resnet_v2_101')

  pred = tf.argmax(end_points_inc_v3['Predictions'], 1)

  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)

  logits = logits_inc_v3
  auxlogits = end_points_inc_v3['AuxLogits']

  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)

  cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                   auxlogits,
                                                   label_smoothing=0.0,
                                                   weights=0.4)
  noise = tf.gradients(cross_entropy, x)[0] 
  noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = tf.nn.conv2d(noise, stack_kernel, [1, 1, 1, 1], padding="SAME")
  noise = momentum * grad + noise
  x = x + alpha * tf.sign(noise)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise

def stop(x, y, i, x_max, x_min, grad):
  return tf.less(i, FLAGS.num_iter)

def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret
# Channel Translation-Invariant Attack
def cim_attack(input_dir, output_dir):

  # some parameter
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, 299, 299, 3]

  #_check_or_create_dir(output_dir)

  with tf.Graph().as_default():
    # Prepare graph
    raw_inputs = tf.placeholder(tf.uint8, shape=[None, 299,299, 3])

    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
   

    y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
    i = tf.constant(0)

    grad = tf.zeros(tf.shape(x_input)) 
    
    x_adv, _, _, _, _, _ = tf.while_loop(stop, non_target_graph, [x_input, y, i, x_max, x_min, grad])
   
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    # s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
    # s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
    # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
    
      s1.restore(sess, model_checkpoint_map['inception_v3'])
      # s2.restore(sess, model_checkpoint_map['inception_v4'])
      # s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
      # s4.restore(sess, model_checkpoint_map['resnet_v2_101'])
    
      for ori_images, filenames, raw_images, true_labels in load_images_with_true_label(input_dir):
        adv_images = sess.run(x_adv, feed_dict={x_input: raw_images})
        save_images(adv_images, filenames, output_dir)
        

if __name__=='__main__':
    time_start=time.time()
    cim_attack(FLAGS.input_dir, FLAGS.output_dir)
    time_end=time.time()
    print('totally cost',time_end-time_start)
    pass
