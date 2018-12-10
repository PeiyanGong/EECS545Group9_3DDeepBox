BIN = 2
OVERLAP = 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
INPUT_SIZE = 224
CLASSES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
BATCH_SIZE = 32

learning_rate = 0.0001
epochs = 50
save_path = './model/'

label_dir = "/home/liu/Desktop/KITTI/training/label_2/"
image_dir = "/home/liu/Desktop/KITTI/data_object_image_2/training/image_2/"
crop_dir = "/home/liu/Desktop/KITTI/crop_2/"

ignore_thresh_truncate = 0.5
ignore_thresh_occlude = 0.5