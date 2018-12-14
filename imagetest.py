import cv2
import numpy as np

noise = 10
image_file = "/home/vincegong/Documents/KITTI3Ddata/3Ddetection/training/image_2/007000.png"
img = cv2.imread(image_file)
img = img.astype(np.float32, copy=False)
height, width = img.shape[:2]
ymax = height+np.random.randint(noise)-(noise/2)
print(height,width)
print(ymax)
print(min(ymax,height))
test = "5."
print(float(test))
print(type(float(test)))