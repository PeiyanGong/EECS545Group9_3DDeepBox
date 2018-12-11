import numpy as np
import tensorflow as tf
from collections import OrderedDict
from config import *
import tensorflow.contrib.slim as slim

DATA_TYPE = tf.float32

def leaky_relu(x, alpha):
	  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def custom_loss(d_train, o_train, b_train, d_pred, o_pred, b_pred):
	# Dim
	loss_d = tf.losses.mean_squared_error(d_train, d_pred)

	# Ori
	masks = tf.reduce_sum(tf.square(o_train), axis=2)
	masks = tf.greater(masks, tf.constant(0.5))
	count = tf.reduce_sum(tf.cast(masks, tf.float32), axis=1)

	dot_prod = (o_train[:,:,0]*o_pred[:,:,0] + o_train[:,:,1]*o_pred[:,:,1])
	loss_o = tf.reduce_mean(tf.reduce_sum((2 - 2 * tf.reduce_mean(dot_prod, axis=0))) / count)

	# masks = tf.greater(b_train, tf.constant(0.5))
	# print(masks.shape)
	# count = tf.reduce_sum(tf.reduce_sum(tf.cast(masks, tf.float32), axis=0), axis=0)
	# dot_prod = (o_train[:,:,0]*o_pred[:,:,0] + o_train[:,:,1]*o_pred[:,:,1])
	# loss_o = 0.
	# for i in range(BIN):
	# 	loss_o += tf.reduce_sum((tf.boolean_mask(1 - dot_prod[:,i], masks[:,i])))
	# loss_o = loss_o / count
	
	loss_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=b_train, logits=b_pred))
	loss = 4. * loss_d + 10. * loss_o + loss_b
	return loss

def VGG_3D(images, training=True):
	# Augmentation
	if training:
		images = tf.image.random_contrast(images, 0.8, 1.2)
		# images = tf.image.random_brightness(images, max_delta=0.1)

	with slim.arg_scope([slim.conv2d, slim.fully_connected],
					  activation_fn=tf.nn.relu,
					  weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
					  weights_regularizer=slim.l2_regularizer(0.0005)):
		net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		net = slim.max_pool2d(net, [2, 2], scope='pool3')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		net = slim.max_pool2d(net, [2, 2], scope='pool5')
		conv5 = tf.contrib.layers.flatten(net)

		dimension = slim.fully_connected(conv5, 512, activation_fn=None, scope='fc7_d')
		dimension = leaky_relu(dimension, 0.1)
		dimension = slim.dropout(dimension, 0.5, scope='dropout7_d', is_training=training)
		dimension = slim.fully_connected(dimension, 3, activation_fn=None, scope='fc8_d')

		orientation = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_o')
		orientation = leaky_relu(orientation, 0.1)
		orientation = slim.dropout(orientation, 0.5, scope='dropout7_o', is_training=training)
		orientation = slim.fully_connected(orientation, BIN*2, activation_fn=None, scope='fc8_o')
		orientation = tf.reshape(orientation, [-1, BIN, 2])
		orientation = tf.nn.l2_normalize(orientation, dim=2)

		confidence = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_b')
		confidence = leaky_relu(confidence, 0.1)
		confidence = slim.dropout(confidence, 0.5, scope='dropout7_b', is_training=training)
		which_bin = slim.fully_connected(confidence, BIN, activation_fn=None, scope='fc8_b')
	   
		confidence = tf.nn.softmax(which_bin)
		
	return dimension, orientation, which_bin, confidence