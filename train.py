import tensorflow as tf
import cv2, os
import numpy as np
import time

from random import shuffle
from data_processing import *
import sys
import argparse
from tqdm import tqdm

from utils import read_stats, prepare_data
from config import *
from network import *

img_in = tf.placeholder(tf.float32, shape = [None, INPUT_SIZE, INPUT_SIZE, 3])
d_label = tf.placeholder(tf.float32, shape = [None, 3])
o_label = tf.placeholder(tf.float32, shape = [None, BIN, 2])
b_label = tf.placeholder(tf.float32, shape = [None, BIN])

dim_stats = read_stats("label_stats.txt")
f = open("label_crop.txt")
label = f.readlines()
num_data = len(label)
train_num = int(np.floor(num_data/BATCH_SIZE))
	
### buile graph
dimension, orientation, bin, _ = VGG_3D(img_in)
loss, loss_d, loss_o, loss_b = custom_loss(d_label, o_label, b_label, dimension, orientation, bin)

# define optimizer
opt_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
# opt_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# Use pretrain VGG
# variables_to_restore = slim.get_variables()[:26] ## vgg16-conv5
# saver = tf.train.Saver(max_to_keep=100)

# Start to train model
for epoch in range(epochs):
	epoch_loss = np.zeros((train_num,1))
	tStart_epoch = time.time()
	batch_loss = 0
	loss_d_, loss_o_, loss_b_ = 0, 0, 0
	random_idx = np.random.permutation(num_data)

	tqdm_iterator = tqdm(range(train_num), ascii=True)
	for num_iters in tqdm_iterator:
		# tqdm_iterator.set_description('Epoch ' + str(epoch+1) + ' : Loss:' + str(batch_loss) + \
		# 	"\nDim Loss : " + str(loss_d_) + " Ori Loss : " + str(loss_o_) + " Bin Loss : " + str(loss_b_))
		tqdm_iterator.set_description('Epoch ' + str(epoch+1) + ' : Loss:' + str(batch_loss))
		curr_idx = random_idx[num_iters*BATCH_SIZE : (num_iters+1)*BATCH_SIZE]
		x_train, d_train, o_train, b_train = prepare_data(label, curr_idx, BATCH_SIZE, dim_stats)
		_, batch_loss, loss_d_, loss_o_, loss_b_ = sess.run([opt_operation, loss, loss_d, loss_o, loss_b], feed_dict={img_in: x_train, d_label: d_train, o_label: o_train, b_label: b_train})

		epoch_loss[num_iters] = batch_loss 

	# save model
	if (epoch+1) % 2 == 0:
		saver.save(sess,save_path+"augment_model", global_step = epoch+1)

	# Print some information
	print ("Epoch:", epoch+1, " done. Loss:", np.mean(epoch_loss))
	tStop_epoch = time.time()
	print ("Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s")
	sys.stdout.flush()