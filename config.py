BIN = 2
OVERLAP = 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
INPUT_SIZE = 224
CLASSES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
BATCH_SIZE = 32

learning_rate = 2e-4
epochs = 50
save_path = './model/'

label_dir = "/home/vincegong/Documents/KITTI3Ddata/3Ddetection/training/label_2/"
image_dir = "/home/vincegong/Documents/KITTI3Ddata/3Ddetection/training/image_2/"
crop_dir = "/home/vincegong/Documents/Course/EECS545/Project/3D-DeepBox-for-EECS545/crop_2_augment/"

ignore_thresh_truncate = 0.5
ignore_thresh_occlude = 1.5