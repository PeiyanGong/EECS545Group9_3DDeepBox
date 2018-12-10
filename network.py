# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import tensorflow as tf
from collections import OrderedDict
from config import *

DATA_TYPE = tf.float32
VARIABLE_COUNTER = 0

def variable(name, shape, initializer,regularizer=None):
	global VARIABLE_COUNTER
	with tf.device('/cpu:0'):
		VARIABLE_COUNTER += np.prod(np.array(shape))
		return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=DATA_TYPE, trainable=True)

def conv_layer(input_tensor,name,kernel_size,output_channels,initializer,stride=1,bn=False,training=False,relu=True, padding='SAME'):
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding=padding)
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		conv_layer = tf.nn.bias_add(conv, biases)
		if bn:
			conv_layer = batch_norm_layer(conv_layer,scope,training)
		if relu:
			conv_layer = tf.nn.relu(conv_layer, name=scope.name)
	print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),conv_layer.get_shape().as_list()))
	return conv_layer

def max_pooling(input_tensor,name,factor=2):
	pool = tf.nn.max_pool(input_tensor, ksize=[1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME', name=name)
	print('Pooling layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),pool.get_shape().as_list()))
	return pool

def batch_norm_layer(input_tensor,scope,training):
	return tf.contrib.layers.batch_norm(input_tensor,scope=scope,is_training=training,decay=0.99)

def dropout_layer(input_tensor,keep_prob,training):
	if training:
		return tf.nn.dropout(input_tensor,keep_prob)
	return input_tensor

def concat_layer(input_tensor1,input_tensor2,axis=3):
	output = tf.concat(3,[input_tensor1,input_tensor2])
	input1_shape = input_tensor1.get_shape().as_list()
	input2_shape = input_tensor2.get_shape().as_list()
	output_shape = output.get_shape().as_list()
	print('Concat layer {0} and {1} -> {2}'.format(input1_shape,input2_shape,output_shape))
	return output

def leaky_relu(x, alpha):
	  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def flatten(input_tensor,name):
	batch_size = input_tensor.get_shape().as_list()[0]
	with tf.variable_scope(name) as scope:
		flat = tf.reshape(input_tensor, [batch_size,-1])
	print('Flatten layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),flat.get_shape().as_list()))
	return flat

def custom_loss(d_train, o_train, b_train, d_pred, o_pred, b_pred):
	loss_d = tf.losses.mean_squared_error(d_train, d_pred)

	masks = tf.reduce_sum(tf.square(o_train), axis=-1)
	masks = tf.greater(masks, tf.constant(0.5))
	count = tf.reduce_sum(tf.cast(masks, tf.float32), axis=1)

	# Define the loss
	loss_o = (o_train[...,0]*o_pred[...,0] + o_train[...,1]*o_pred[...,1])
	loss_o = tf.reduce_mean(tf.reduce_sum((2 - 2 * tf.reduce_mean(loss_o,axis=0))) / count)
	
	loss_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=b_train, logits=b_pred))
	loss = 2. * loss_d + 5. * loss_o + loss_b
	return loss, loss_d, loss_o, loss_b

def VGG_3D(images, training=True):
	print('-'*30)
	print('Network Architecture')
	print('-'*30)
	global VARIABLE_COUNTER
	VARIABLE_COUNTER = 0
	layer_name_dict = {}
	def layer_name(base_name):
		if base_name not in layer_name_dict:
			layer_name_dict[base_name] = 0
		layer_name_dict[base_name] += 1
		name = base_name + str(layer_name_dict[base_name])
		return name

	bn = True
	# initializer = tf.contrib.layers.variance_scaling_initializer()
	initializer = tf.truncated_normal_initializer(0.0, 0.01)

	dw_h_convs = OrderedDict()

	# Augmentation
	if training:
		images = tf.image.random_contrast(images, 0.75, 1.25)
		images = tf.image.random_brightness(images, max_delta=0.5)
	# Process Input
	dw_h_convs[0] = conv_layer(images, layer_name('conv'), 3, 64, initializer=initializer, bn = bn, training = training)
	dw_h_convs[0] = conv_layer(dw_h_convs[0], layer_name('conv'), 3, 64, initializer=initializer, bn = bn, training = training)
	dw_h_convs[0] = max_pooling(dw_h_convs[0], 'pool1')

	dw_h_convs[1] = conv_layer(dw_h_convs[0], layer_name('conv'), 3, 128, initializer=initializer, bn = bn, training = training)
	dw_h_convs[1] = conv_layer(dw_h_convs[1], layer_name('conv'), 3, 128, initializer=initializer, bn = bn, training = training)
	# dw_h_convs[1] = tf.concat([dw_h_convs[1], dw_h_convs[0]], axis=-1)
	dw_h_convs[1] = max_pooling(dw_h_convs[1], 'pool2')

	dw_h_convs[2] = conv_layer(dw_h_convs[1], layer_name('conv'), 3, 256, initializer=initializer, bn = bn, training = training)
	dw_h_convs[2] = conv_layer(dw_h_convs[2], layer_name('conv'), 3, 256, initializer=initializer, bn = bn, training = training)
	dw_h_convs[2] = conv_layer(dw_h_convs[2], layer_name('conv'), 3, 256, initializer=initializer, bn = bn, training = training)
	# dw_h_convs[2] = tf.concat([dw_h_convs[2], dw_h_convs[1]], axis = -1)
	dw_h_convs[2] = max_pooling(dw_h_convs[2], 'pool3')

	dw_h_convs[3] = conv_layer(dw_h_convs[2], layer_name('conv'), 3, 512, initializer=initializer, bn = bn, training = training)
	dw_h_convs[3] = conv_layer(dw_h_convs[3], layer_name('conv'), 3, 512, initializer=initializer, bn = bn, training = training)
	dw_h_convs[3] = conv_layer(dw_h_convs[3], layer_name('conv'), 3, 512, initializer=initializer, bn = bn, training = training)
	# dw_h_convs[3] = tf.concat([dw_h_convs[3], dw_h_convs[2]], axis=-1)
	dw_h_convs[3] = max_pooling(dw_h_convs[3], 'pool4')

	dw_h_convs[4] = conv_layer(dw_h_convs[3], layer_name('conv'), 3, 512, initializer=initializer, bn = bn, training = training)
	dw_h_convs[4] = conv_layer(dw_h_convs[4], layer_name('conv'), 3, 512, initializer=initializer, bn = bn, training = training)
	dw_h_convs[4] = conv_layer(dw_h_convs[4], layer_name('conv'), 3, 512, initializer=initializer, bn = bn, training = training)
	# dw_h_convs[4] = tf.concat([dw_h_convs[4], dw_h_convs[3]], axis=-1)

	# dw_h_convs[5] = conv_layer(dw_h_convs[4], layer_name('conv'), 3, 512, initializer=initializer, bn = bn, training = training)
	# dw_h_convs[5] = conv_layer(dw_h_convs[5], layer_name('conv'), 3, 512, initializer=initializer, bn = bn, training = training)
	# dw_h_convs[5] = conv_layer(dw_h_convs[5], layer_name('conv'), 3, 512, initializer=initializer, bn = bn, training = training)
	dw_h_convs[5] = max_pooling(dw_h_convs[4], 'pool5')

	fc = tf.contrib.layers.flatten(dw_h_convs[4])
	dimension = tf.layers.dense(fc, 512, name='dense_d1')
	dimension = leaky_relu(dimension, 0.1)
	dimension = tf.nn.dropout(dimension, 0.5)
	dimension = tf.layers.dense(dimension, 3, name='dense_d2')

	orientation = tf.layers.dense(fc, 256, name='dense_o1')
	orientation = leaky_relu(orientation, 0.1)
	orientation = tf.nn.dropout(orientation, 0.5)
	orientation = tf.layers.dense(orientation, 2*BIN, name='dense_o2')
	orientation = tf.reshape(orientation, [-1, BIN, 2])
	orientation = tf.nn.l2_normalize(orientation, dim=2)
	
	bin = tf.layers.dense(fc, 256, name='dense_b1')
	bin = leaky_relu(bin, 0.1)
	bin = tf.nn.dropout(bin, 0.5)
	bin = tf.layers.dense(bin, BIN, name='dense_b2')
	classfication = tf.nn.softmax(bin)

	return dimension, orientation, bin, classfication
