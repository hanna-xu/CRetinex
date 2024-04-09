import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from msia_BN_3_M import *
import random
from utils import *


def lrelu(x, trainbable=None):
	return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
	with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
		pool_size = 2
		deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable=True)
		deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1],
										name=scope_name)
		deconv_output = tf.concat([deconv, x2], 3)
		deconv_output.set_shape([None, None, None, output_channels * 2])
		return deconv_output

def DecomNet(input):
	with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
		with tf.variable_scope('reflectance', reuse=tf.AUTO_REUSE):
			conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
			pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding='SAME')
			conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
			pool2 = slim.max_pool2d(conv2, [2, 2], stride=2, padding='SAME')
			conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
			up8 = upsample_and_concat(conv3, conv2, 64, 128, 'g_up_1')
			conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
			up9 = upsample_and_concat(conv8, conv1, 32, 64, 'g_up_2')
			conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
			conv10 = slim.conv2d(conv9, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
			R_out = tf.sigmoid(conv10)

		with tf.variable_scope('illumination', reuse=tf.AUTO_REUSE):
			l_conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='l_conv1_2')
			l_conv3 = tf.concat([l_conv2, conv9], 3)
			l_conv4 = slim.conv2d(l_conv3, 1, [1, 1], rate=1, activation_fn=None, scope='l_conv1_4')
			L_out = tf.sigmoid(l_conv4)
	return R_out, L_out

def NoiseNet(input):  # is_training=True
	with tf.variable_scope('NoiseNet', reuse=tf.AUTO_REUSE):
		conv1 = slim.conv2d(input, 32, [9, 9], rate=1, activation_fn=lrelu, scope='g_conv1')
		conv2 = slim.conv2d(conv1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2')

		res1 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='res1_conv1')
		res1 = slim.conv2d(res1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='res1_conv2')
		res1 = tf.add(res1, tf.identity(conv2))

		res2 = slim.conv2d(res1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='res2_conv1')
		res2 = slim.conv2d(res2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='res2_conv2')
		res2 = tf.add(res2, tf.identity(res1))

		res3 = slim.conv2d(res2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='res3_conv1')
		res3 = slim.conv2d(res3, 64, [3, 3], rate=1, activation_fn=lrelu, scope='res3_conv2')
		res3 = tf.add(res3, tf.identity(res2))

		conv2_ds1 = slim.conv2d(conv2, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='g_conv2_ds1')
		conv2_ds2 = slim.conv2d(conv2_ds1, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='g_conv2_ds2')
		conv2_ds3 = slim.conv2d(conv2_ds2, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='g_conv2_ds3')

		# attention
		x = conv2_ds3
		x_mp = tf.expand_dims(tf.reduce_max(x, axis=-1), axis=-1)
		x_ap = tf.expand_dims(tf.reduce_mean(x, axis=-1), axis=-1)
		x_attention_map = tf.sigmoid(
			slim.conv2d(tf.concat([x_mp, x_ap], axis=-1), 1, [7, 7], stride=1, activation_fn=None,
						scope='ds_attention_conv'))
		x = x * tf.tile(x_attention_map, [1, 1, 1, 64])

		us_x1 = tf.layers.conv2d_transpose(inputs=x, filters=64, kernel_size=(6, 6), strides=(2, 2), padding='same')
		us_x1 = slim.conv2d(us_x1, 64, [5, 5], stride=1, activation_fn=lrelu, scope='us_x_conv1')

		us_x2 = tf.layers.conv2d_transpose(inputs=us_x1, filters=64, kernel_size=(6, 6), strides=(2, 2), padding='same')
		us_x2 = slim.conv2d(us_x2, 64, [5, 5], stride=1, activation_fn=lrelu, scope='us_x_conv2')

		us_x3 = tf.layers.conv2d_transpose(inputs=us_x2, filters=64, kernel_size=(6, 6), strides=(2, 2), padding='same')
		us_x3 = slim.conv2d(us_x3, 64, [5, 5], stride=1, activation_fn=lrelu, scope='us_x_conv3')

		conv4 = slim.conv2d(tf.concat([res3, us_x3, conv1], axis=-1), 96, [5, 5], rate=1, activation_fn=lrelu, scope='g_conv4')
		conv_mp = tf.expand_dims(tf.reduce_max(conv4, axis=-1), axis=-1)
		conv_ap = tf.expand_dims(tf.reduce_mean(conv4, axis=-1), axis=-1)
		conv_attention_map = tf.sigmoid(
			slim.conv2d(tf.concat([conv_mp, conv_ap], axis=-1), 1, [7, 7], stride=1, activation_fn=None,
						scope='attention_conv'))
		conv4 = conv4 * tf.tile(conv_attention_map, [1, 1, 1, 96])

		conv5= tf.concat([conv4, conv1], axis=-1)
		res4 = slim.conv2d(conv5, 128, [3, 3], rate=1, activation_fn=lrelu, scope='res4_conv1')
		res4 = slim.conv2d(res4, 128, [3, 3], rate=1, activation_fn=lrelu, scope='res4_conv2')
		res4 = tf.add(res4, tf.identity(conv5))

		conv6 = slim.conv2d(res4, 3, [5, 5], rate=1, activation_fn=None, scope='g_conv6')
		conv6 = tf.nn.tanh(conv6)

		noise = conv6
	return noise, conv2, x, conv4

def ColorNet(input):
	with tf.variable_scope('ColorNet', reuse=tf.AUTO_REUSE):
		c_conv1 = slim.conv2d(input, 32, [5, 5], activation_fn=lrelu, scope='g_conv1')
		c_conv2 = slim.conv2d(c_conv1, 64, [3, 3], activation_fn=lrelu, scope='g_conv2')

		res1 = slim.conv2d(c_conv2, 64, [3, 3], activation_fn=lrelu, scope='res1_conv1')
		res1 = slim.conv2d(res1, 64, [3, 3], activation_fn=lrelu, scope='res1_conv2')
		res1 = tf.add(res1, tf.identity(c_conv2))

		res2 = slim.conv2d(res1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='res2_conv1')
		res2 = slim.conv2d(res2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='res2_conv2')
		res2 = tf.add(res2, tf.identity(res1))

		c_conv3 = slim.conv2d(res2, 128, [5, 5], stride=[2, 2], activation_fn=lrelu, scope='g_conv3')
		c_conv4 = slim.conv2d(c_conv3, 64, [5, 5], stride=[2, 2], activation_fn=lrelu, scope='g_conv4')
		c_conv5 = slim.conv2d(c_conv4, 64, [5, 5], stride=[2, 2], activation_fn=lrelu, scope='g_conv5')
		fc1 = tf.reduce_mean(c_conv5, axis=[1, 2])
		fc1 = slim.fully_connected(fc1, num_outputs=32, activation_fn=None, scope='fc1')
		fc1 = tf.maximum(fc1, fc1 * 0.2)
		fc2 = slim.fully_connected(fc1, num_outputs=3, activation_fn=None, scope='fc2')
		estimate = tf.sigmoid(fc2)
		estimate = tf.nn.l2_normalize(estimate, dim=1)
	return estimate


def Illumination_adjust_net(input_i, input_r):
	input = tf.concat([input_i, input_r], axis=-1)
	with tf.variable_scope('I_adjust_Net', reuse=tf.AUTO_REUSE):

		channel = 32
		kernel_size=3

		conv1 = tf.layers.conv2d(input, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
		conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
		conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)

		up1 = tf.layers.conv2d_transpose(inputs=conv3, filters=channel, kernel_size=(6, 6), strides=(2, 2), padding='same')
		deconv1 = tf.layers.conv2d(tf.concat([up1, conv2], axis=-1), channel, 5, padding='same', activation=tf.nn.relu)
		deconv2 = tf.layers.conv2d(deconv1, channel, 3, padding='same', activation=tf.nn.relu)

		up2 = tf.layers.conv2d_transpose(inputs=deconv2, filters=channel, kernel_size=(6, 6), strides=(2, 2), padding='same')
		deconv3 = tf.layers.conv2d(tf.concat([up2, conv1], axis=-1), channel, 5, padding='same', activation=tf.nn.relu)
		deconv4 = tf.layers.conv2d(deconv3, channel, 3, padding='same', activation=tf.nn.relu)

		up3 = tf.layers.conv2d_transpose(inputs=deconv4, filters=channel, kernel_size=(6, 6), strides=(2, 2),
										 padding='same')
		deconv5 = tf.layers.conv2d(tf.concat([up3, input], axis=-1), channel, 5, padding='same', activation=tf.nn.relu)
		deconv6 = tf.layers.conv2d(tf.concat([deconv5, apply_gamma_tf(input_i), input_i], axis=-1), channel, 3, padding='same', activation=tf.nn.relu)

		feature_fusion1 = tf.layers.conv2d(deconv6, channel, 3, padding='same', activation=tf.nn.relu)
		feature_fusion3 = tf.layers.conv2d(feature_fusion1, channel, 1, padding='same', activation=None)
		output = tf.layers.conv2d(feature_fusion3, 1, 1, padding='same', activation=None)
		L_enhance = tf.clip_by_value(output, 0, 1)
	return L_enhance