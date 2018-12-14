import tensorflow as tf
import cv2, os
import numpy as np
from random import shuffle
import glob
from config import *
from tqdm import tqdm

def parse_annotation(image_process=False):
	label_files = sorted(glob.glob(label_dir + "*.txt"))[0:7000]
	label_crop = open("label_crop.txt", "w")
	# sum of h, w, l and count
	dims_count = np.zeros((len(CLASSES), 4))
	tqdm_iterator = tqdm(label_files, ascii=True)
	for label_file in tqdm_iterator:
		tqdm_iterator.set_description(label_file)
		f = open(label_file, 'r')
		f_lines = f.readlines()
		count = 0
		image_file = image_dir + label_file.split('/')[-1].split('.')[0] + ".png"
		for line in f_lines:
			image_crop = crop_dir + label_file.split('/')[-1].split('.')[0] + "_" + str(count) + ".png"
			line = line.split(' ')
			truncated = np.abs(float(line[1]))
			occluded  = np.abs(float(line[2]))
			alpha = float(line[3])
			if line[0] in CLASSES:
				if truncated > ignore_thresh_truncate or occluded > ignore_thresh_occlude:
					continue

				new_alpha = alpha + np.pi/2.
				# change alpha in [0. 2PI]
				if new_alpha < 0:
					new_alpha = new_alpha + 2.*np.pi
				new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi) 

				xmin = int(float(line[4]))
				ymin = int(float(line[5]))
				xmax = int(float(line[6]))
				ymax = int(float(line[7]))
				# xmin = int(float(line[4]))
				# ymin = int(float(line[5]))
				# xmax = int(float(line[6]))
				# ymax = int(float(line[7]))

				info = image_crop + " " + line[0] + " " \
						+ line[8] + " " + line[9] + " " + line[10] + " " + str(new_alpha) + "\n"
				label_crop.write(info)
				
				if (image_process):
					img = cv2.imread(image_file)
					img_new = img[ymin:ymax+1, xmin:xmax+1]
					img_new = cv2.resize(img_new, (INPUT_SIZE, INPUT_SIZE))
					cv2.imwrite(image_crop, img_new)
				
				for i in range(3):
					dims_count[CLASSES.index(line[0]), i] += float(line[8 + i])
				dims_count[CLASSES.index(line[0]), 3] += 1
				count += 1

	label_stats = open("label_stats.txt", "w")
	for i in range(len(CLASSES)):
		dims_count[i, :] /= dims_count[i, 3]
		info = CLASSES[i] + " " + str(dims_count[i, 0]) + " " + str(dims_count[i, 1]) + " " + str(dims_count[i, 2]) + "\n"
		label_stats.write(info)

	label_stats.close()
	label_crop.close()

if __name__ == "__main__":
	parse_annotation(image_process=True)