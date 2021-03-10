# coding=UTF-8   
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
import sys
from nets import inception_v3, inception_v4,inception_resnet_v2,resnet_v2
import argparse
import skimage
from skimage import io
from six.moves import xrange
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
slim = tf.contrib.slim
from tensorflow import reduce_sum
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# 声明一些攻击参数
CHECKPOINTS_DIR = '../models'
model_checkpoint_map = {
    'inception_resnet_v2': os.path.join(CHECKPOINTS_DIR,'inception_resnet_v2.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(CHECKPOINTS_DIR,'ens_adv_inception_resnet_v2_rename'),
    'ens3_adv_inception_v3': os.path.join(CHECKPOINTS_DIR,'ens3_adv_inception_v3_rename'),
    'inception_v4': os.path.join(CHECKPOINTS_DIR, 'inception_v4.ckpt'),
    'inception_v3': os.path.join(CHECKPOINTS_DIR, 'inception_v3.ckpt'),
    'ens4_adv_inception_v3': os.path.join(CHECKPOINTS_DIR, 'ens4_adv_inception_v3_rename')}

parser = argparse.ArgumentParser(description='EnsembleFool')
parser.add_argument('--input_dir', type=str, default='../ImageNet/images')
parser.add_argument('--output_dir', type=str, default='../dim-out/e4inc3')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_epsilon', type=int, default=16)
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--momentum', type=float, default=1.0)
FLAGS = parser.parse_args()


# load images
def load_images_with_true_label(input_dir):
    ori_images = []
    images = []
    filenames = []
    true_labels = []
    idx = 0
    label = pd.read_csv(os.path.join('../ImageNet', 'labels.csv'))
    filename2label = {label.iloc[i]['ImageId'] : label.iloc[i]['TrueLabel'] for i in range(len(label))}
    for filename in filename2label.keys():
        ori_image = imread(os.path.join(FLAGS.input_dir, filename), mode='RGB').astype(np.float)
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
        image = imresize(image, [299, 299])
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')


def graph(x, y, yi, i, x_max, x_min, grad, pred_w):  #[x_input, adv_x, y, i, x_max, x_min, grad]
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  num_iter = FLAGS.num_iter
  alpha = eps / num_iter
  momentum = FLAGS.momentum
  num_classes = 1001

  image = x

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_inc_v3, end_points_inc_v3 = inception_v3.inception_v3(
        image, num_classes=num_classes, is_training=False, scope='InceptionV3')

  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    logits_inc_v4, end_points_inc_v4 = inception_v4.inception_v4(
        image, num_classes=num_classes, is_training=False, scope='InceptionV4')

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_inc_res_v2, end_points_inc_res_v2 = inception_resnet_v2.inception_resnet_v2(
        image, num_classes=num_classes, is_training=False, scope='InceptionResnetV2')

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
      image, num_classes=num_classes, is_training=False, scope='ens_adv_inception_resnet_v2')

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens3_inc_v3, end_points_ens3_inc_v3 = inception_v3.inception_v3(
        image, num_classes=num_classes, is_training=False, scope='ens3_adv_inception_v3')
  
  # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
  #   logits_ens4_inc_v3, end_points_ens4_inc_v3 = inception_v3.inception_v3(
  #       image, num_classes=num_classes, is_training=False, scope='ens4_adv_inception_v3')
  
  pred = tf.argmax(end_points_inc_v3['Predictions']+ \
    end_points_inc_v4['Predictions']+end_points_inc_res_v2['Predictions']+ \
    end_points_ensadv_res_v2['Predictions']+end_points_ens3_inc_v3['Predictions'], 1)

  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  
  logits = (logits_inc_v3*pred_w[0]+logits_inc_v4*pred_w[1]+logits_inc_res_v2*pred_w[2] + \
      logits_ensadv_res_v2*pred_w[3]+logits_ens3_inc_v3*pred_w[4])/tf.reduce_sum(pred_w)

  auxlogits = (end_points_inc_v3['AuxLogits']*pred_w[0]+end_points_inc_v4['AuxLogits']*pred_w[1]+ \
      end_points_inc_res_v2['AuxLogits']*pred_w[2]+end_points_ensadv_res_v2['AuxLogits']*pred_w[3]+\
      end_points_ens3_inc_v3['AuxLogits']*pred_w[4])/(tf.reduce_sum(pred_w))
  logits = logits + auxlogits*0.4
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
  noise = momentum * grad + noise
  x = x + alpha * tf.sign(noise)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)

  inc3_pred = tf.concat([[end_points_inc_v3['Predictions'][i,yi[i]] for i in range(yi.shape[0])]],axis=0)
  inc4_pred = tf.concat([[end_points_inc_v4['Predictions'][i,yi[i]] for i in range(yi.shape[0])]],axis=0)
  res2_pred = tf.concat([[end_points_inc_res_v2['Predictions'][i,yi[i]] for i in range(yi.shape[0])]],axis=0)
  ens_res2_pred = tf.concat([[end_points_ensadv_res_v2['Predictions'][i,yi[i]] for i in range(yi.shape[0])]],axis=0)
  ens3_inc3_pred = tf.concat([[end_points_ens3_inc_v3['Predictions'][i,yi[i]] for i in range(yi.shape[0])]],axis=0)
  # ens4_inc3_pred = tf.concat([[end_points_ens4_inc_v3['Predictions'][i,yi[i]] for i in range(yi.shape[0])]],axis=0)
  pred_list =  tf.concat([inc3_pred,inc4_pred,res2_pred,ens_res2_pred,ens3_inc3_pred],axis=0)
  return x, y, yi, i, x_max, x_min, noise, pred_list

def stop(x, y, yi, i, x_max, x_min, grad, pred):
  return tf.less(i, FLAGS.num_iter)


def ensemblefool(input_dir, output_dir):

  # some parameter
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, 299, 299, 3]

  #_check_or_create_dir(output_dir)

  with tf.Graph().as_default():
    # Prepare graph

    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
   

    y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
    y1 = tf.placeholder(tf.int64, shape=[FLAGS.batch_size])
    i = tf.constant(0)
    pred = tf.ones([5],tf.float32)
    grad = tf.zeros(tf.shape(x_input)) 

    x_adv, _, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, y1, i, x_max, x_min, grad, pred])
   
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='ens_adv_inception_resnet_v2'))
    s4 = tf.train.Saver(slim.get_model_variables(scope='ens3_adv_inception_v3'))
    # s5 = tf.train.Saver(slim.get_model_variables(scope='ens4_adv_inception_v3'))
    s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      s1.restore(sess, model_checkpoint_map['inception_v4'])
      s2.restore(sess, model_checkpoint_map['inception_resnet_v2'])
      s3.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
      s4.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
      # s5.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
      s6.restore(sess, model_checkpoint_map['inception_v3'])
    
      for ori_images, filenames, raw_images, true_labels in load_images_with_true_label(input_dir):
        adv_images = sess.run(x_adv, feed_dict={x_input: raw_images,y1:true_labels})
        save_images(adv_images, filenames, output_dir)
        

if __name__=='__main__':
    ensemblefool(FLAGS.input_dir, FLAGS.output_dir)
    pass
