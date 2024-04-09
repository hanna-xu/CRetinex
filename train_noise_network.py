# coding: utf-8
from __future__ import print_function
import os, time, random
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import *
from model import *
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch size')
parser.add_argument('--train_data_dir', dest='train_data_dir', default='./dataset/train',
					help='directory for training inputs')
parser.add_argument('--train_result_dir', dest='train_result_dir', default='./eval_result/noise/',
					help='directory for noise net training results')

args = parser.parse_args()

batch_size = args.batch_size
patch_size = args.patch_size

eps = 1e-8
tau = 0.5

# define loss
def mutual_i_input_loss(input_I_low, input_im):
	input_gray = tf.image.rgb_to_grayscale(input_im)
	low_gradient_x = gradient(input_I_low, "x")
	input_gradient_x = gradient(input_gray, "x")
	x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))
	low_gradient_y = gradient(input_I_low, "y")
	input_gradient_y = gradient(input_gray, "y")
	y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
	mut_loss = tf.reduce_mean(x_loss + y_loss)
	return mut_loss

def rgb2gray(r, g, b):
	return 0.2989 * r + 0.587 * g + 0.114 * b

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

mask = mask(p_R_low, p_R_high, patch_size)

estimate_low1 = ColorNet(p_R_low)
estimate_high1 = ColorNet(p_R_high)
p_R_low_wo_C = apply_correction(p_R_low, estimate_low1)
p_R_high_wo_C = apply_correction(p_R_high, estimate_high1)
Z_low1, _, _, _ = NoiseNet(p_R_low_wo_C)
Z_high1, _, _, _ = NoiseNet(p_R_high_wo_C)
R_low1 = p_R_low_wo_C - Z_low1
R_high1 =  p_R_high_wo_C - Z_high1

_, F1_low, F2_low, F3_low = NoiseNet(R_low1)
_, F1_high, F2_high, F3_high = NoiseNet(p_R_high)


estimate_low_wo_C = tf.reduce_sum(tf.reshape(p_R_low_wo_C, shape=[batch_size, -1, 3]) * tf.reshape(mask, shape=[batch_size, -1, 3]), axis=1)
estimate_low_wo_C = tf.nn.l2_normalize(estimate_low_wo_C, dim=1)

estimate_high = tf.reduce_sum(tf.reshape(p_R_high, shape=[batch_size, -1, 3]) * tf.reshape(mask, shape=[batch_size, -1, 3]), axis=1)
estimate_high = tf.nn.l2_normalize(estimate_high, dim=1)

loss_grad = tf.reduce_mean(tf.abs(grad_RGB(R_low1) - grad_RGB(p_R_high)))
loss_l2 = tf.reduce_mean(tf.square(R_low1 - p_R_high))
loss_dis = tf.reduce_mean(tf.square(F1_low - F1_high)) + tf.reduce_mean(tf.square(F2_low - F2_high))
loss_denoise = loss_l2 + 0.2 * loss_grad + 0.0001 * loss_dis

###
lr = tf.placeholder(tf.float32, name='learning_rate')
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

var_Decom_I = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_Decom_Z = [var for var in tf.trainable_variables() if 'NoiseNet' in var.name]
var_Decom_C = [var for var in tf.trainable_variables() if 'ColorNet' in var.name]

train_op_denoise = optimizer.minimize(loss_denoise, var_list=var_Decom_Z)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list=var_Decom_I + var_Decom_C + var_Decom_Z, max_to_keep=5)
print("[*] Initialize model successfully...")

train_loss = loss_denoise


# load data
###train_data
train_low_data = []
train_high_data = []
train_low_data_names = glob(args.train_data_dir + '/low/*.png')
train_low_data_names.sort()
train_high_data_names = glob(args.train_data_dir + '/high/*.png')
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

epoch = 600
learning_rate = 0.0001

sample_dir = args.train_result_dir
if not os.path.isdir(sample_dir):
	os.makedirs(sample_dir)

eval_every_epoch = 100
train_phase = 'noise estimation:'
numBatch = len(train_low_data) // int(batch_size)

print('loaded ' + './checkpoint/color_net/')
saver_IC = tf.train.Saver(var_list=var_Decom_I + var_Decom_C)
ckpt_illu_color = tf.train.get_checkpoint_state('./checkpoint/color_net/')
saver_IC.restore(sess, ckpt_illu_color.model_checkpoint_path)

checkpoint_dir = './checkpoint/noise_net/'
if not os.path.isdir(checkpoint_dir):
	os.makedirs(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
	print('loaded ' + ckpt.model_checkpoint_path)
	strs = ckpt.model_checkpoint_path.split('.')[-2]
	iter_num = int(strs.split('/')[-1])
	print(iter_num)
	saver1 = tf.train.Saver(var_list=var_Decom_I + var_Decom_C + var_Decom_Z)
	saver1.restore(sess, ckpt.model_checkpoint_path)
else:
	print('No noisenet pretrained model!')
	iter_num = 0

writer = tf.compat.v1.summary.FileWriter("logs/noise_net/", sess.graph)

start_step = 0
start_epoch = 0

print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0

NUM = 3
input_low_tb = tf.compat.v1.summary.image('a_input_low', input_low, max_outputs=NUM)
input_high_tb = tf.compat.v1.summary.image('a_input_high', input_high, max_outputs=NUM)
I_low_tb = tf.compat.v1.summary.image('c_illu_low', output_I_low, max_outputs=NUM)
I_high_tb = tf.compat.v1.summary.image('c_illu_high', output_I_high, max_outputs=NUM)
p_R_low_tb = tf.compat.v1.summary.image('p_reflectance_low', p_R_low, max_outputs=NUM)
R_low_wo_C_tb = tf.compat.v1.summary.image('p_reflectance_low_wo_C', p_R_low_wo_C, max_outputs=NUM)
R_low_tb = tf.compat.v1.summary.image('reflectance_low', R_low1, max_outputs=NUM)
R_high_tb = tf.compat.v1.summary.image('p_reflectance_high', p_R_high, max_outputs=NUM)
mask_tb = tf.compat.v1.summary.image('mask', mask, max_outputs=NUM)

Z_low_tb = tf.compat.v1.summary.image('noise_low', Z_low1, max_outputs=NUM)
Z_high_tb = tf.compat.v1.summary.image('noise_high', Z_high1, max_outputs=NUM)

loss1 = tf.compat.v1.summary.scalar('loss_l2',  loss_l2)
loss2 = tf.compat.v1.summary.scalar('loss_dis',  loss_dis)
loss3 = tf.compat.v1.summary.scalar('loss_grad',  loss_grad)
loss4 = tf.compat.v1.summary.scalar('loss_denoise',  loss_denoise)

merge_summary = tf.compat.v1.summary.merge(
	[input_low_tb, input_high_tb, mask_tb, p_R_low_tb, R_low_tb, R_high_tb, I_low_tb, I_high_tb, R_low_wo_C_tb,
	 Z_low_tb, Z_high_tb, loss1, loss2, loss3, loss4])

for epoch in range(start_epoch, epoch):
	for batch_id in range(start_step, numBatch):
		batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
		batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")

		for patch_id in range(batch_size):
			h, w, _ = train_low_data[image_id].shape
			x = random.randint(0, h - patch_size)
			y = random.randint(0, w - patch_size)
			rand_mode = random.randint(0, 7)
			batch_input_low[patch_id, :, :, :] = data_augmentation(
				train_low_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
			batch_input_high[patch_id, :, :, :] = data_augmentation(
				train_high_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
			image_id = (image_id + 1) % len(train_low_data)
			if image_id == 0:
				tmp = list(zip(train_low_data, train_high_data))
				random.shuffle(tmp)
				train_low_data, train_high_data = zip(*tmp)

		_, loss = sess.run([train_op_denoise, train_loss], feed_dict={input_low: batch_input_low, \
															  input_high: batch_input_high, lr: learning_rate})

		iter_num += 1
		if iter_num % 100==0:
			result = sess.run(merge_summary, feed_dict={input_low: batch_input_low, input_high: batch_input_high, lr: learning_rate})
			writer.add_summary(result, iter_num)
			writer.flush()

			print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
				  % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))

	if (epoch+1) % eval_every_epoch == 0:
		print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
		for idx in range(len(eval_low_data)):
			input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
			input_low_eval = np.tile(input_low_eval, [batch_size, 1, 1, 1])
			result_0, result_1, result_2, result_3= sess.run([input_low, p_R_low, p_R_low_wo_C, R_low1],
															  feed_dict={input_low: input_low_eval})
			result_low = np.concatenate(
				[result_0[0:1, :, :, :], result_1[0:1, :, :, :], result_2[0:1, :, :, :], result_3[0:1, :, :, :]],
				 axis = 2)

			input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
			input_high_eval = np.tile(input_high_eval, [batch_size, 1, 1, 1])
			result_00, result_11, result_22, result_33 = sess.run([input_high, p_R_high, p_R_high_wo_C, R_high1],
				feed_dict={input_high: input_high_eval})
			result_high = np.concatenate(
				[result_00[0:1, :, :, :], result_11[0:1, :, :, :], result_22[0:1, :, :, :], result_33[0:1, :, :, :]],
				axis=2)
			save_images(os.path.join(sample_dir, '%d_%d.png' % (idx + 1, epoch + 1)), result_low, result_high)

	if (epoch +1) % 5 ==0:
		saver.save(sess, checkpoint_dir + str(iter_num) + '.ckpt')

print("[*] Finish training for phase %s." % train_phase)
