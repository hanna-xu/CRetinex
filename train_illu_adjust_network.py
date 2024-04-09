# coding: utf-8
from __future__ import print_function
import os
import time
import random
#from skimage import color
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob

batch_size = 8
patch_size = 128
tau = 0.1
def apply_correction(img, estimate):
	norm = tf.expand_dims(tf.reduce_prod(estimate, axis=1), axis=-1)
	norm = tf.tile(norm ** (1.0 / 3), [1, 3])
	out = norm[:, None, None, :] * img / (1e-6 + estimate[:, None, None, :])
	corrected_img = tf.clip_by_value(out, 0, 1.0)
	return corrected_img

def rgb2gray(rgb):
	return 0.2989 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]

def rgb2gray_tf(rgb):
	return 0.2989 * rgb[:, :, :,0:1] + 0.587 * rgb[:, :,:,1:2] + 0.114 * rgb[:, :,:,2:3]

def SSIM(img1, img2, size = 11, sigma = 1.5):
	window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
	k1 = 0.01
	k2 = 0.03
	L = 1  # depth of image (255 in case the image has a different scale)
	c1 = (k1 * L) ** 2
	c2 = (k2 * L) ** 2
	mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu1_sq = mu1 * mu1
	mu2_sq = mu2 * mu2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_sq
	sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
	sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2
	ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
	value = tf.reduce_mean(ssim_map, axis = [1, 2, 3])
	return value

def similarity_eva(x, y):
	return tf.reduce_mean(-tf.square(x-y), axis=[1, 2, 3])

def _tf_fspecial_gauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function"""
	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
	x_data = np.expand_dims(x_data, axis = -1)
	x_data = np.expand_dims(x_data, axis = -1)
	y_data = np.expand_dims(y_data, axis = -1)
	y_data = np.expand_dims(y_data, axis = -1)
	x = tf.constant(x_data, dtype = tf.float32)
	y = tf.constant(y_data, dtype = tf.float32)
	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
	return g / tf.reduce_sum(g)

sess = tf.Session()

input_low = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='input_low')
input_high = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='input_high')


[_, I_low] = DecomNet(input_low)
[_, I_high] = DecomNet(input_high)

I_low_3 = tf.concat([I_low, I_low, I_low], axis=3)
I_high_3 = tf.concat([I_high, I_high, I_high], axis=3)
output_I_low = I_low_3
output_I_high = I_high_3
p_R_low = tf.clip_by_value(input_low/I_low_3, 0, 1)
p_R_high = tf.clip_by_value(input_high/I_high_3, 0, 1)

estimate_low = ColorNet(p_R_low)
p_R_low_wo_C = apply_correction(p_R_low, estimate_low)
Z_low, _, _, _ = NoiseNet(p_R_low_wo_C)
R_low = p_R_low_wo_C - Z_low

I_low_adjust = Illumination_adjust_net(I_low, R_low)
I_low_adjust_3 = tf.concat([I_low_adjust, I_low_adjust, I_low_adjust], axis=3)
adjusted_low = tf.clip_by_value(I_low_adjust_3 * R_low, 0, 1.0)


loss_square = tf.reduce_mean(tf.square(I_low_adjust - I_high))
loss_ssim = tf.reduce_mean(1 - tf_ssim(I_low_adjust, I_high))
loss_image = tf.reduce_mean(1 - tf_ssim(rgb2gray_tf(adjusted_low), rgb2gray_tf(input_high)))
loss_adjust = loss_ssim + 0.2 * loss_image

loss_sim = tf.reduce_mean(tf.abs(I_low_adjust - I_high)) + tf.reduce_mean(tf.square(grad_gray(I_low_adjust)-grad_gray(I_high)))# tf.reduce_mean(1 - SSIM(I_low_adjust, I_high)) #

lr = tf.placeholder(tf.float32, name='learning_rate')
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

var_Decom_I = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_Decom_Z = [var for var in tf.trainable_variables() if 'NoiseNet' in var.name]
var_Decom_C = [var for var in tf.trainable_variables() if 'ColorNet' in var.name]
var_adjust = [var for var in tf.trainable_variables() if 'I_adjust_Net' in var.name]

var_Decom = var_Decom_I + var_Decom_C + var_Decom_Z
saver_adjust = tf.train.Saver(var_list=var_adjust +var_Decom)
saver_Decom = tf.train.Saver(var_list = var_Decom)
train_op_adjust = optimizer.minimize(loss_adjust, var_list = var_adjust)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")

# load data
###train_data
train_low_data = []
train_high_data = []
train_low_data_names = glob('./dataset/train' + '/low/*.png')
train_low_data_names.sort()
train_high_data_names = glob('./dataset/train' + '/high/*.png')
train_high_data_names.sort()
assert len(train_low_data_names) == len(train_high_data_names)
print('[*] Number of training data: %d' % len(train_low_data_names))
for idx in range(len(train_low_data_names)):
	low_im = load_images(train_low_data_names[idx])
	train_low_data.append(low_im)
	high_im = load_images(train_high_data_names[idx])
	train_high_data.append(high_im)

###eval_data
eval_low_data = []
eval_high_data = []
eval_low_data_name = glob('./dataset/eval/low/*.png')
eval_low_data_name.sort()
eval_high_data_name = glob('./dataset/eval/high/*.png*')
eval_high_data_name.sort()
for idx in range(len(eval_low_data_name)):
	eval_low_im = load_images(eval_low_data_name[idx])
	eval_low_data.append(eval_low_im)
	eval_high_im = load_images(eval_high_data_name[idx])
	eval_high_data.append(eval_high_im)


learning_rate = 0.0001
epoch = 1000
eval_every_epoch = 10
train_phase = 'adjustment'
numBatch = len(train_low_data) // int(batch_size)
train_op = train_op_adjust
train_loss = loss_adjust
saver = saver_adjust

print('loaded ' + './checkpoint/noise_net/')
saver_Decom = tf.train.Saver(var_list=var_Decom)
ckpt_Decom = tf.train.get_checkpoint_state('./checkpoint/noise_net/')
saver_Decom.restore(sess, ckpt_Decom.model_checkpoint_path)

checkpoint_dir = './checkpoint/illu_adjust/'
if not os.path.isdir(checkpoint_dir):
	os.makedirs(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
	print('loaded ' + ckpt.model_checkpoint_path)
	strs = ckpt.model_checkpoint_path.split('.')[-2]
	iter_num = int(strs.split('/')[-1])
	print(iter_num)
	saver1 = tf.train.Saver(var_list=var_adjust)
	saver1.restore(sess, ckpt.model_checkpoint_path)
else:
	print('No illumination adjustment network pretrained model!')
	iter_num = 0

writer = tf.compat.v1.summary.FileWriter("logs/illu_adjust/", sess.graph)

NUM = 3
input_low_tb = tf.compat.v1.summary.image('a_input_low', input_low, max_outputs=NUM)
input_high_tb = tf.compat.v1.summary.image('a_input_high', input_high, max_outputs=NUM)
I_low_tb = tf.compat.v1.summary.image('c_illu_low', output_I_low, max_outputs=NUM)
I_high_tb = tf.compat.v1.summary.image('c_illu_high', output_I_high, max_outputs=NUM)
I_low_adjust_tb = tf.compat.v1.summary.image('c_illu_low_adjust', I_low_adjust, max_outputs=NUM)
p_R_low_tb = tf.compat.v1.summary.image('p_reflectance_low', p_R_low, max_outputs=NUM)
R_low_tb = tf.compat.v1.summary.image('reflectance_low', R_low, max_outputs=NUM)
R_high_tb = tf.compat.v1.summary.image('p_reflectance_high', p_R_high, max_outputs=NUM)
adjust_low_tb = tf.compat.v1.summary.image('adjusted_low', adjusted_low, max_outputs=NUM)

loss1 = tf.compat.v1.summary.scalar('loss_adjust', loss_adjust)
loss2 = tf.compat.v1.summary.scalar('loss_ssim', loss_ssim)
loss3 = tf.compat.v1.summary.scalar('loss_image', loss_image)

merge_summary = tf.compat.v1.summary.merge(
	[input_low_tb, input_high_tb, p_R_low_tb, R_low_tb, R_high_tb, I_low_tb, I_high_tb, I_low_adjust_tb, adjust_low_tb,
	 loss1, loss2, loss3])

start_step = 0
start_epoch = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

sample_dir = './train_illu_adjust_result/'
if not os.path.isdir(sample_dir):
	os.makedirs(sample_dir)

start_time = time.time()
image_id = 0

for epoch in range(start_epoch, epoch):
	for batch_id in range(start_step, numBatch):
		batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
		batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
		gamma_num = 0
		for patch_id in range(batch_size):
			h, w, _ = train_low_data[image_id].shape
			x = random.randint(0, h - patch_size)
			y = random.randint(0, w - patch_size)
			rand_mode = random.randint(0, 7)
			batch_input_low[patch_id, :, :, :] = data_augmentation(
				train_low_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
			one_batch_input_high = data_augmentation(
				train_high_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
			if np.mean(rgb2gray(one_batch_input_high)) < 0.2:
				gamma_num += 1
				batch_input_high[patch_id, :, :, :] = apply_gamma(one_batch_input_high)
			else:
				batch_input_high[patch_id, :, :, :] = one_batch_input_high

			image_id = (image_id + 1) % len(train_low_data)
			if image_id == 0:
				tmp = list(zip(train_low_data, train_high_data))
				random.shuffle(tmp)
				train_low_data, train_high_data = zip(*tmp)
		_, loss = sess.run([train_op, train_loss], feed_dict={input_low: batch_input_low, \
															  input_high: batch_input_high,
															  lr: learning_rate})

		iter_num += 1
		if iter_num % 100 == 0:
			result = sess.run(merge_summary,
							  feed_dict={input_low: batch_input_low, input_high: batch_input_high, lr: learning_rate})
			writer.add_summary(result, iter_num)
			writer.flush()
			print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
				  % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))

	if (epoch + 1) % eval_every_epoch == 0:
		print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
		for idx in range(len(eval_low_data)):
			input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
			input_low_eval = np.tile(input_low_eval, [batch_size, 1, 1, 1])
			input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
			input_high_eval = np.tile(input_high_eval, [batch_size, 1, 1, 1])
			result_0, result_1, result_2, result_3, result_4, result_5 = sess.run(
				[input_low, I_low_3, R_low, I_low_adjust_3, adjusted_low, input_high],
				feed_dict={input_low: input_low_eval, input_high: input_high_eval})
			result_first = np.concatenate(
				[result_0[0:1, :, :, :], result_1[0:1, :, :, :], result_2[0:1, :, :, :]], axis=2)
			result_second = np.concatenate(
				[result_3[0:1, :, :, :], result_4[0:1, :, :, :], result_5[0:1, :, :, :]], axis=2)

			save_images(os.path.join(sample_dir, '%d_%d.png' % (idx + 1, epoch + 1)), result_first, result_second)

	if (epoch + 1) % 5 == 0:
		saver.save(sess, checkpoint_dir + str(iter_num) + '.ckpt')


print("[*] Finish training for phase %s." % train_phase)



