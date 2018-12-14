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

# x_train, d_train, o_train, b_train = prepare_data(label, range(8), BATCH_SIZE, dim_stats)
# for i in range(train_img.shape[0]):
#     cv2.imwrite("gt_test_2/" + str(i) + ".png", x_train[i,:])
# print(d_train, o_train, b_train)

### buile graph
dimension, orientation, bin, _ = VGG_3D(img_in)
loss = custom_loss(d_label, o_label, b_label, dimension, orientation, bin)

# define optimizer
opt_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# opt_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
# Use pretrain VGG
variables_to_restore = tf.contrib.slim.get_variables()[:26]
saver = tf.train.Saver()

ckpt_list = tf.contrib.framework.list_variables('./vgg_16.ckpt')[1:-7]
new_ckpt_list = []
for name in range(1,len(ckpt_list),2):
	tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name-1][0]: variables_to_restore[name]})
	tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name][0]: variables_to_restore[name-1]})
sess.run(tf.global_variables_initializer())

filename = "loss.txt"
f = open(filename,"w")

# Start to train model
for epoch in range(epochs):
	epoch_loss = np.zeros((train_num,1))
	tStart_epoch = time.time()
	batch_loss = 0
	random_idx = np.random.permutation(num_data)

	tqdm_iterator = tqdm(range(train_num), ascii=True)
	for num_iters in tqdm_iterator:
		tqdm_iterator.set_description('Epoch ' + str(epoch+1) + ' : Loss:' + str(batch_loss))
		curr_idx = random_idx[num_iters*BATCH_SIZE : (num_iters+1)*BATCH_SIZE]
		x_train, d_train, o_train, b_train = prepare_data(label, curr_idx, BATCH_SIZE, dim_stats)
		_, batch_loss = sess.run([opt_operation, loss], feed_dict={img_in: x_train, d_label: d_train, o_label: o_train, b_label: b_train})

		epoch_loss[num_iters] = batch_loss 

	# save model
	if (epoch+1) % 2 == 0:
		saver.save(sess,save_path+"augment", global_step = epoch+1)
	# save loss
	f.write(" ".join(map(str, epoch_loss.flatten())))
	f.write("\n")
	# Print some information
	print ("Epoch:", epoch+1, " done. Loss:", np.mean(epoch_loss))
	tStop_epoch = time.time()
	print ("Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s")
	sys.stdout.flush()