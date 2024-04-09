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
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--train_data_dir', dest='train_data_dir', default='./dataset/train',
					help='directory for training inputs')
parser.add_argument('--train_result_dir', dest='train_result_dir', default='./train_decom_result/',
					help='directory for decomnet training results')

args = parser.parse_args()

batch_size = args.batch_size
patch_size = args.patch_size

eps=1e-8

# define loss

def mutual_i_loss(input_I_low, input_I_high):
	low_gradient_x = gradient(input_I_low, "x")
	high_gradient_x = gradient(input_I_high, "x")
	x_loss = (low_gradient_x + high_gradient_x) * tf.exp(-10 * (low_gradient_x + high_gradient_x))
	low_gradient_y = gradient(input_I_low, "y")
	high_gradient_y = gradient(input_I_high, "y")
	y_loss = (low_gradient_y + high_gradient_y) * tf.exp(-10 * (low_gradient_y + high_gradient_y))
	mutual_loss = tf.reduce_mean(x_loss + y_loss)
	return mutual_loss


def mutual_i_input_loss(input_I_low, input_im):
	input_gray = tf.image.rgb_to_grayscale(input_im)
	low_gradient_x = gradient(input_I_low, "x")
	input_gradient_x = gradient(input_gray, "x")
	x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(tf.abs(input_gradient_x), 0.01)))
	low_gradient_y = gradient(input_I_low, "y")
	input_gradient_y = gradient(input_gray, "y")
	y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(tf.abs(input_gradient_y), 0.01)))
	mut_loss = tf.reduce_mean(x_loss + y_loss)
	return mut_loss

def apply_correction(img, estimate):
	norm = tf.expand_dims(tf.reduce_prod(estimate, axis=1), axis=-1)
	norm = tf.tile(norm ** (1.0 / 3), [1, 3])
	out = norm[:, None, None, :] * img / (1e-6 + estimate[:, None, None, :])
	corrected_img = tf.clip_by_value(out, 0, 1.0)
	return corrected_img

sess = tf.Session()

input_low = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='input_low')
input_high = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='input_high')

[R_low, I_low] = DecomNet(input_low)
[R_high, I_high] = DecomNet(input_high)

I_low_3 = tf.concat([I_low, I_low, I_low], axis=3)
I_high_3 = tf.concat([I_high, I_high, I_high], axis=3)

# network output
output_R_low = R_low
output_R_high = R_high
output_I_low = I_low_3
output_I_high = I_high_3

recon_img_low = output_R_low * output_I_low
recon_img_high = output_R_high * output_I_high

input_low_level = tf.reduce_mean(input_low, axis=[1, 2, 3])
input_high_level = tf.reduce_mean(input_high, axis=[1, 2, 3])
recon_loss_low = tf.reduce_mean(tf.reduce_mean(tf.abs(recon_img_low - input_low), axis=[1, 2, 3])/ (input_low_level+eps))
recon_loss_high = tf.reduce_mean(tf.reduce_mean(tf.abs(recon_img_high - input_high), axis=[1, 2, 3]) / (input_high_level+eps))

equal_R_loss = tf.reduce_mean(tf.abs(output_R_low- output_R_high))

i_input_mutual_loss_high = mutual_i_input_loss(I_high, input_high)
i_input_mutual_loss_low = mutual_i_input_loss(I_low, input_low)

loss_Decom_I = recon_loss_high + 0.3 * recon_loss_low \
			 + 0.8 * equal_R_loss + 0.5 * i_input_mutual_loss_high + 0.5 * i_input_mutual_loss_low

lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
# var_reflect = [var for var in tf.trainable_variables() if 'reflectance' in var.name]
# var_illum = [var for var in tf.trainable_variables() if 'illumination' in var.name]

train_op_Decom = optimizer.minimize(loss_Decom_I, var_list=var_Decom)
sess.run(tf.global_variables_initializer())

saver_Decom = tf.train.Saver(var_list=var_Decom, max_to_keep=5)
print("[*] Initialize model successfully...")

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

epoch = 2000
learning_rate = 0.0001

sample_dir = args.train_result_dir
if not os.path.isdir(sample_dir):
	os.makedirs(sample_dir)

eval_every_epoch = 100
train_phase = 'decomposition'
numBatch = len(train_low_data) // int(batch_size)
train_op = train_op_Decom
train_loss = loss_Decom_I
saver = saver_Decom

checkpoint_dir = './checkpoint/decomposition/'
if not os.path.isdir(checkpoint_dir):
	os.makedirs(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
	print('loaded ' + ckpt.model_checkpoint_path)
	strs = ckpt.model_checkpoint_path.split('.')[-2]
	iter_num = int(strs.split('/')[-1])
	print(iter_num)
	saver1 = tf.train.Saver(var_list=var_Decom)
	saver1.restore(sess, ckpt.model_checkpoint_path)
else:
	print('No decomnet pretrained model!')
	iter_num = 0

writer = tf.compat.v1.summary.FileWriter("logs/decomposition/", sess.graph)
start_step = 0
start_epoch = 0

print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0

NUM = 3
input_low_tb = tf.compat.v1.summary.image('a_input_low', input_low, max_outputs=NUM)
input_high_tb = tf.compat.v1.summary.image('a_input_high', input_high, max_outputs=NUM)
R_low_tb = tf.compat.v1.summary.image('b_reflectance_low', output_R_low, max_outputs=NUM)
R_high_tb = tf.compat.v1.summary.image('b_reflectance_high', output_R_high, max_outputs=NUM)
I_low_tb = tf.compat.v1.summary.image('c_illu_low', output_I_low, max_outputs=NUM)
I_high_tb = tf.compat.v1.summary.image('c_illu_high', output_I_high, max_outputs=NUM)

loss1 = tf.compat.v1.summary.scalar('recon_loss_high', recon_loss_high)
loss2 = tf.compat.v1.summary.scalar('recon_loss_low', recon_loss_low)
loss3 = tf.compat.v1.summary.scalar('R_loss', equal_R_loss)
loss5 = tf.compat.v1.summary.scalar('i_input_high', i_input_mutual_loss_high)
loss6 = tf.compat.v1.summary.scalar('i_input_low', i_input_mutual_loss_low)

merge_summary = tf.compat.v1.summary.merge(
	[input_low_tb, input_high_tb, R_low_tb, R_high_tb, I_low_tb, I_high_tb, loss1, loss2, loss3, loss5, loss6])

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

		_, loss = sess.run([train_op, train_loss], feed_dict={input_low: batch_input_low, \
															  input_high: batch_input_high, \
															  lr: learning_rate})

		iter_num += 1
		if iter_num % 50==0:
			result = sess.run(merge_summary, feed_dict={input_low: batch_input_low, input_high: batch_input_high, lr: learning_rate})
			writer.add_summary(result, iter_num)
			writer.flush()

			print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
				  % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))

	if (epoch + 1) % eval_every_epoch == 0:
		print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
		for idx in range(len(eval_low_data)):
			input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
			input_low_eval = np.tile(input_low_eval, [batch_size, 1, 1, 1])
			result_0, result_1, result_2, result_3, result_4 = sess.run([input_low, recon_img_low, output_I_low, input_low/(output_I_low+eps), output_R_low],
															  feed_dict={input_low: input_low_eval})
			result_low = np.concatenate(
				[result_0[0:1, :, :, :], result_1[0:1, :, :, :], result_2[0:1, :, :, :], np.clip(result_3[0:1, :, :, :], 0, 1), result_4[0:1, :, :, :]],
				 axis = 2)

			input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
			input_high_eval = np.tile(input_high_eval, [batch_size, 1, 1, 1])
			result_00, result_11, result_22, result_33, result_44 = sess.run(
				[input_high, recon_img_high, output_I_high, input_high/(output_I_high+eps), output_R_high],
				feed_dict={input_high: input_high_eval})
			result_high = np.concatenate(
				[result_00[0:1, :, :, :], result_11[0:1, :, :, :], result_22[0:1, :, :, :], np.clip(result_33[0:1, :, :, :], 0, 1), result_44[0:1, :, :, :]],
				axis=2)
			save_images(os.path.join(sample_dir, '%d_%d.png' % (idx + 1, epoch + 1)), result_low, result_high)

	saver.save(sess, checkpoint_dir + str(iter_num) + '.ckpt')

print("[*] Finish training for phase %s." % train_phase)
