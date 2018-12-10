import tensorflow as tf
import cv2, os
import numpy as np
from random import shuffle
import glob
from config import *

def compute_anchors(angle):
	anchors = []
	
	wedge = 2.*np.pi/BIN
	l_index = int(angle/wedge)
	r_index = l_index + 1
	
	if (angle - l_index*wedge) < wedge/2 * (1+OVERLAP/2):
		anchors.append([l_index, angle - l_index*wedge])
		
	if (r_index*wedge - angle) < wedge/2 * (1+OVERLAP/2):
		anchors.append([r_index%BIN, angle - r_index*wedge])
		
	return anchors

def prepare_data(labels, indices, batch_size, dim_stats):
	x_train = []
	d_train = np.zeros((batch_size, 3))
	o_train = np.zeros((batch_size, BIN, 2))
	b_train = np.zeros((batch_size, BIN))
	for i in range(batch_size):
		label_line = labels[indices[i]]
		label = label_line.split(" ")
		c_idx = CLASSES.index(label[1])
		dim_stat = dim_stats[c_idx]
		# read image
		img = cv2.imread(label[0])

		# read label
		d_train[i, :] = np.array(list(map(float, label[2:5]))) - dim_stat
		angle = float(label[5])

		flip = np.random.rand(1)
		if flip[0] < 0.5: # DO NOT flip
			anchors = compute_anchors(angle)
			for anchor in anchors:
				o_train[i, anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
				b_train[i, anchor[0]] = 1.
		else:
			anchors = compute_anchors(2*np.pi - angle)
			img = cv2.flip(img, 1)
			for anchor in anchors:
				o_train[i, anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
				b_train[i, anchor[0]] = 1.

		x_train.append(img)
		b_train[i, :] = b_train[i, :] / np.sum(b_train[i, :])

	return np.array(x_train), d_train, o_train, b_train

def read_stats(stats_file):
	f = open(stats_file, "r")
	f_line = f.readlines()
	dim_stats = []
	for line in f_line:
		data = line.split(" ")
		dim_stat = np.zeros(3)
		for i in range(3):
			dim_stat[i] = float(data[i + 1])
		dim_stats.append(dim_stat)

	return dim_stats

if __name__ == "__main__":
	dim_stats = read_stats("label_stats.txt")

	f = open("label_crop.txt")
	label = f.readlines()
	indices = range(16)
	x_train, d_train, o_train, b_train = prepare_data(label, indices, 16, dim_stats)

	print(x_train.shape, d_train.shape, o_train.shape, b_train.shape)
	