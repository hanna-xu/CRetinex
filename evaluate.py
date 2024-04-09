# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color,filters
import argparse
import scipy.io as scio

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='', help='dataset for testing')
parser.add_argument('--save_dir', dest='save_dir', default='./test_results/LOL/', help='directory for testing outputs')
# parser.add_argument('--test_dir', dest='test_dir', default='/data/xh/competitors/KinD_plus/test_images/AGLIE/low/', help='directory for testing inputs')
# parser.add_argument('--test_dir', dest='test_dir', default='/data/xh/competitors/KinD_plus/LOLdataset/our485/low/', help='directory for testing inputs')
parser.add_argument('--test_dir', dest='test_dir', default='/data/xh/DeepRetinex++/test_images/LOL/low/', help='directory for testing inputs')
parser.add_argument('--adjustment', dest='adjustment', default=False, help='whether to adjust illumination')
# parser.add_argument('--ratio', dest='ratio', default=5.0, help='ratio for illumination adjustment')

args = parser.parse_args()

sess = tf.Session()
input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')

[_, I_low] = DecomNet(input_low)
I_low_3 = tf.concat([I_low, I_low, I_low], axis=3)
p_R_low = tf.clip_by_value(input_low/I_low_3, 0, 1)
estimate_low = ColorNet(p_R_low)
p_R_low_wo_C = apply_correction(p_R_low, estimate_low)
Z_low, _, _, _ = NoiseNet(p_R_low_wo_C)
R_low = p_R_low_wo_C - Z_low

# I_low_norm = illu_norm(I_low, 1)
I_low_adjust = Illumination_adjust_net(I_low, R_low)
I_low_adjust_3 = tf.concat([I_low_adjust, I_low_adjust, I_low_adjust], axis=3)
adjusted_low = I_low_adjust_3 * R_low #tf.clip_by_value(I_low_adjust_3 * R_low, 0, 1.0)

# load pretrained model
var_Decom_I = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_Decom_Z = [var for var in tf.trainable_variables() if 'NoiseNet' in var.name]
var_Decom_C = [var for var in tf.trainable_variables() if 'ColorNet' in var.name]
var_adjust = [var for var in tf.trainable_variables() if 'I_adjust_Net' in var.name]

var_all = var_Decom_I + var_Decom_C + var_Decom_Z + var_adjust
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_all += bn_moving_vars

'''abla'''
# saver = tf.train.Saver(var_list = var_all)
# checkpoint_dir ='./checkpoint/illu_adjust/'
# ckpt_pre=tf.train.get_checkpoint_state(checkpoint_dir)
# if ckpt_pre:
#     print('loaded '+ckpt_pre.model_checkpoint_path)
#     saver.restore(sess,ckpt_pre.model_checkpoint_path)
# else:
#     print('No checkpoint!')
saver1 = tf.train.Saver(var_list = var_Decom_I + var_Decom_C + var_adjust +bn_moving_vars)
checkpoint_dir ='./checkpoint/illu_adjust/'
ckpt_pre=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver1.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No checkpoint!')
saver2 = tf.train.Saver(var_list = var_Decom_Z)
checkpoint_dir ='./checkpoint/decom_net_noise/'
ckpt_pre=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver2.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No checkpoint!')



###load eval data
eval_low_data = []
eval_img_name =[]
Height = []
Width = []
eval_low_data_name = glob(args.test_dir+args.dataset+'/'+'*')
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im, height, width = load_images_H_W(eval_low_data_name[idx])
    eval_low_data.append(eval_low_im)
    Height.append(height)
    Width.append(width)

sample_dir = args.save_dir + args.dataset + '/'
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)
if not os.path.isdir(sample_dir + 'reflectance/'):
    os.makedirs(sample_dir + 'reflectance/')
if not os.path.isdir(sample_dir + 'illumination/'):
    os.makedirs(sample_dir + 'illumination/')
if not os.path.isdir(sample_dir + 'illumination/adjusted/'):
    os.makedirs(sample_dir + 'illumination/adjusted/')
if not os.path.isdir(sample_dir + 'noise/'):
    os.makedirs(sample_dir + 'noise/')


print("Start evalating!")
Time = np.zeros([len(eval_low_data), 1])
for idx in range(len(eval_low_data)):
    start_time = time.time()
    name = eval_img_name[idx]
    print("\033[1;33;40m", args.dataset, "\t[", idx+1, "/", len(eval_low_data), "]: ", name, "\033[0m")
    input_low_eval = eval_low_data[idx]
    input_low_eval = np.expand_dims(input_low_eval, axis=0)

    enhanced_result = sess.run(adjusted_low[0, :, :, :], feed_dict={input_low: input_low_eval})
    end_time = time.time()
    decom_r_low, decom_i_low, decom_z_low, adjusted_i_low, p_r_low_without_c, p_r_low = sess.run(
        [R_low[0, :, :, :], I_low[0, :, :, :], Z_low[0, :, :, :], I_low_adjust[0, :, :, :], p_R_low_wo_C[0, :, :, :],
        p_R_low[0, :, :, :]], feed_dict={input_low: input_low_eval})
    save_images(os.path.join(sample_dir, '%s.jpg' % (name)),
                cv2.resize(enhanced_result, (Width[idx], Height[idx]), interpolation=cv2.INTER_AREA))
    # save_images(os.path.join(sample_dir + 'reflectance/', '%s.png' % (name)),
    #             cv2.resize(decom_r_low, (Width[idx], Height[idx]), interpolation=cv2.INTER_AREA))
    # save_images(os.path.join(sample_dir + 'illumination/', '%s.png' % (name)),
    #             cv2.resize(decom_i_low, (Width[idx], Height[idx]), interpolation=cv2.INTER_AREA))
    save_images(os.path.join(sample_dir + 'reflectance_w_z/', '%s.png' % (name)),
                cv2.resize(p_r_low_without_c, (Width[idx], Height[idx]), interpolation=cv2.INTER_AREA))
    save_images(os.path.join(sample_dir + 'reflectance_w_z_and_c/', '%s.png' % (name)),
                cv2.resize(p_r_low, (Width[idx], Height[idx]), interpolation=cv2.INTER_AREA))
    save_images(os.path.join(sample_dir + 'illumination/adjusted', '%s.png' % (name)),
                cv2.resize(adjusted_i_low, (Width[idx], Height[idx]), interpolation=cv2.INTER_AREA))


    Time[idx] = end_time - start_time
scio.savemat(sample_dir+'time.mat',{'T':Time})
    
