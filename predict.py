import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2, os
import numpy as np
import time
from random import shuffle
from data_processing import *
import sys
import argparse
from tqdm import tqdm
from utils import read_stats, prepare_data
from network import *

#####


NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']

save_path = './model/'

dims_avg = {'Cyclist': np.array([ 1.73532436,  0.58028152,  1.77413709]), 'Van': np.array([ 2.18928571,  1.90979592,  5.07087755]), 'Tram': np.array([  3.56092896,   2.39601093,  18.34125683]), 'Car': np.array([ 1.52159147,  1.64443089,  3.85813679]), 'Pedestrian': np.array([ 1.75554637,  0.66860882,  0.87623049]), 'Truck': np.array([  3.07392252,   2.63079903,  11.2190799 ])}


#### Placeholder
inputs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
d_label = tf.placeholder(tf.float32, shape = [None, 3])
o_label = tf.placeholder(tf.float32, shape = [None, BIN, 2])
c_label = tf.placeholder(tf.float32, shape = [None, BIN])


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='3D bounding box')
    parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'test')
    parser.add_argument('--image',dest = 'image',help='Image path')
    parser.add_argument('--label',dest = 'label',help='Label path')
    parser.add_argument('--box2d',dest = 'box2d',help='2D detection path')
    parser.add_argument('--output',dest = 'output',help='Output path', default = './validation/result_2/')
    parser.add_argument('--model',dest = 'model')
    parser.add_argument('--gpu',dest = 'gpu',default= '0')
    args = parser.parse_args()

    return args

def test(model, image_dir, box2d_loc, box3d_loc):
    dim_stats = read_stats("label_stats.txt")
    ### buile graph
    dimension, orientation, _, confidence = VGG_3D(inputs, training=False)

    ### GPU config 
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, model)

    # create a folder for saving result
    if os.path.isdir(box3d_loc) == False:
        os.mkdir(box3d_loc)

    # Load image & run testing 
    # USE 7000 to 7480 to eval
    all_image = sorted(os.listdir(image_dir))[7000:]

    for f in all_image:
        image_file = image_dir + f
        box2d_file = box2d_loc + f.replace('png', 'txt')
        box3d_file = box3d_loc + f.replace('png', 'txt')
        print (image_file)
        with open(box3d_file, 'w') as box3d:
            img = cv2.imread(image_file)
            img = img.astype(np.float32, copy=False)

            for line in open(box2d_file):
                line = line.strip().split(' ')
                truncated = np.abs(float(line[1]))
                occluded  = np.abs(float(line[2]))

                obj = {'xmin':int(float(line[4])),
                       'ymin':int(float(line[5])),
                       'xmax':int(float(line[6])),
                       'ymax':int(float(line[7])),
                       }

                patch = img[obj['ymin']:obj['ymax'],obj['xmin']:obj['xmax']]
                # patch = cv2.resize(patch, (NORM_H, NORM_W))
                patch = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE))
                # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)
                # patch = patch - np.array([[[103.939, 116.779, 123.68]]])
                patch = np.expand_dims(patch, 0)
                prediction = sess.run([dimension, orientation, confidence], feed_dict={inputs: patch})
                # Transform regressed angle
                max_anc = np.argmax(prediction[2][0])
                anchors = prediction[1][0][max_anc]

                if anchors[1] > 0:
                    angle_offset = np.arccos(anchors[0])
                else:
                    angle_offset = -np.arccos(anchors[0])

                wedge = 2.*np.pi/BIN
                angle_offset = angle_offset + max_anc*wedge
                angle_offset = angle_offset % (2.*np.pi)

                angle_offset = angle_offset - np.pi/2
                if angle_offset > np.pi:
                    angle_offset = angle_offset - (2.*np.pi)

                line[3] = str(angle_offset)
                 
                line[-1] = angle_offset + np.arctan(float(line[11]) / float(line[13]))
                
                # Transform regressed dimension
                if line[0] in CLASSES:
                    dims = dim_stats[CLASSES.index(line[0])] + prediction[0][0]
                else:
                    dims = dim_stats[CLASSES.index('Car')] + prediction[0][0]

                line = line[:8] + list(dims) + line[11:]
                
                # Write regressed 3D dim and oritent to file
                line = ' '.join([str(item) for item in line]) +' '+ str(np.max(prediction[2][0]))+ '\n'
                box3d.write(line)

 

if __name__ == "__main__":
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.image is None:
        raise IOError(('Image not found.'.format(args.image)))
    if args.box2d is None :
        raise IOError(('2D bounding box not found.'.format(args.box2d)))
    else:
        if args.model is None:
            raise IOError(('Model not found.'.format(args.model)))

        test(args.model, args.image, args.box2d, args.output)

