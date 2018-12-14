import cv2
import numpy as np
import config

file_index = "007469"
font = cv2.FONT_HERSHEY_SIMPLEX
pad_size = 0

def drawLine(img, index1, index2, points, color, thickness):
    cv2.line(img, (points[index1][0], points[index1][1]), (points[index2][0], points[index2][1]), color=color, thickness=thickness)
    return img

def project3Dto2D(xyz, hwl, theta, img, intrinsic):
    points_3D = []
    points_2D = []
    
    for i in [-0.5, 0.5]:
        for j in [-0.5, 0.5]:
            for k in [0, -1]:
                p = np.array([i*hwl[2], k*hwl[0], j*hwl[1]])
                points_3D.append(p)

    rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    for p in points_3D:
        pr = np.matmul(rotation, p)
        pr = pr + np.array(xyz)
        pr = np.append(pr, 1)
        pr = np.matmul(intrinsic, pr.T)
        p = pr/pr[2]
        p = np.int_(p)
        points_2D.append(p)
    points_2D = np.array(points_2D)

    return [np.min(points_2D[:, 0]), np.min(points_2D[:, 1]), np.max(points_2D[:, 0]), np.max(points_2D[:, 1])]
    
def project3Dto2DDraw(xyz, hwl, theta, img, intrinsic, color, thickness=1):
    points_3D = []
    points_2D = []
    
    for i in [-0.5, 0.5]:
        for j in [-0.5, 0.5]:
            for k in [0, -1]:
                p = np.array([i*hwl[2], k*hwl[0], j*hwl[1]])
                points_3D.append(p)

    rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    for p in points_3D:
        pr = np.matmul(rotation, p)
        pr = pr + np.array(xyz)
        pr = np.append(pr, 1)
        pr = np.matmul(intrinsic, pr.T)
        p = pr/pr[2] + np.array([pad_size, pad_size, 0])
        p = np.int_(p)
        points_2D.append(p)
    pt2Darr = np.array(points_2D)
    # print(np.max(pt2Darr[:,0]) - np.min(pt2Darr[:,0]), np.max(pt2Darr[:,1]) - np.min(pt2Darr[:,1]))
    for order in line_order:
        img = drawLine(img, order[0], order[1], points_2D, color, thickness)

    return img

line_order = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [2, 6], [4, 6], [0, 4], [1, 3], [7, 3], [7, 5], [5, 1]]
image_file = '/home/vincegong/Documents/KITTI3Ddata/3Ddetection/training/image_2/' + file_index + ".png"
# image_file = "data_object_image_2/testing/image_2/" + file_index + ".png"
img = cv2.imread(image_file)
intrinsic = np.zeros((3, 4))
h = img.shape[0]
w = img.shape[1]

# get calibration parameters
calib_file = '/home/vincegong/Documents/KITTI3Ddata/3Ddetection/training/calib/' + file_index + '.txt'
# calib_file = 'data_object_calib/testing/calib/' + file_index + '.txt'
with open(calib_file) as f:
    data = f.readlines()
    line = data[2].split(" ")
    for i in range(3):
        for j in range(4):
            intrinsic[i,j] = float(line[i*4 + j + 1])

c_u = intrinsic[0,2]
c_v = intrinsic[1,2]
f_u = intrinsic[0,0]
f_v = intrinsic[1,1]
b_x = intrinsic[0,3]/(-f_u)
b_y = intrinsic[1,3]/(-f_v)
# print (c_u, c_v, b_x, b_y)
# print(intrinsic)

# get ground truth
# label_file = '/home/vincegong/Documents/KITTI3Ddata/3Ddetection/training/label_2/' + file_index + '.txt'
# label_file = '/home/vincegong/Documents/Course/EECS545/Project/3D-DeepBox-for-EECS545/output_orign/' + file_index + '.txt'
# label_file = '/home/vincegong/Documents/Course/EECS545/Project/3D-DeepBox-for-EECS545/output_adam/' + file_index + '.txt'
label_file = '/home/vincegong/Documents/Course/EECS545/Project/3D-DeepBox-for-EECS545/output_final/' + file_index + '.txt'
with open(label_file) as f:
    bbox2Ds = []
    bbox3Ds = []
    
    data = f.readlines()
    for line in data:
        attr = line.split(" ")
        if attr[0] == "DontCare":
            continue
        box2D = attr[4:8]
        box2D = list(map(float, box2D))
        box2D = list(map(int, box2D))
        bbox2Ds.append(box2D)

        box3D = attr[8:15]
        box3D.append(attr[3])
        box3D.append(attr[1])
        # 0:3 -> h,w,l    3:6 -> x,y,z  6 -> theta  7 -> alpha  8 -> truncate
        box3D = list(map(float, box3D))
        bbox3Ds.append(box3D)

img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
# draw 3D bounding box
for i in range(len(bbox2Ds)):
    bbox2D = bbox2Ds[i]
    bbox3D = bbox3Ds[i]

    cv2.line(img, (bbox2D[0], bbox2D[1]), (bbox2D[0], bbox2D[3]), color=(0, 255, 0))
    cv2.line(img, (bbox2D[0], bbox2D[3]), (bbox2D[2], bbox2D[3]), color=(0, 255, 0))
    cv2.line(img, (bbox2D[2], bbox2D[3]), (bbox2D[2], bbox2D[1]), color=(0, 255, 0))
    cv2.line(img, (bbox2D[2], bbox2D[1]), (bbox2D[0], bbox2D[1]), color=(0, 255, 0))
    center =  np.array([(bbox2D[0] + bbox2D[2])/2, (bbox2D[1] + bbox2D[3])/2, 1])
    depth = 1
    depth_min = 0.5
    depth_max = 1000
    # print(bbox3D[8], bbox2D[2] - bbox2D[0], bbox2D[3] - bbox2D[1])
    h_bb = (bbox2D[3] - bbox2D[1])*(bbox2D[2] - bbox2D[0])
    # h_bb = bbox2D[3] - bbox2D[1]

    while True:
        depth = (depth_min + depth_max)/2.0
        x = (center[0] - c_u)*depth/f_u + b_x
        y = (center[1] - c_v)*depth/f_v + b_y
        p = np.array([x, y + bbox3D[0]/2, depth])
        theta = np.arctan2(x, depth) + bbox3D[7]
        projected_box = project3Dto2D(p, bbox3D[0:3], theta, img, intrinsic)
        h_p = (projected_box[3] - projected_box[1])*(projected_box[2] - projected_box[0])
        # h_p = min(projected_box[3], h) - max(projected_box[1], 0)

        if (abs(h_bb - h_p) < h_bb*0.05): # converges
            center = center - np.array([(projected_box[2] + projected_box[0])/2 - center[0], (projected_box[3] + projected_box[1])/2 - center[1], 1])
            x = (center[0] - c_u)*depth/f_u + b_x
            y = (center[1] - c_v)*depth/f_v + b_y
            p = np.array([x, y + bbox3D[0]/2, depth])
            break
        elif (h_bb > h_p):
            depth_max = depth
        else:
            depth_min = depth

    print(bbox3D[0:3], bbox3D[7])
    img = project3Dto2DDraw(p, bbox3D[0:3], theta, img, intrinsic, (255, 255, 0))
    # img = project3Dto2DDraw(bbox3D[3:6], bbox3D[0:3], bbox3D[6], img, intrinsic, (255, 0, 255))
    
    cv2.putText(img, str(bbox3D[7]),(projected_box[0] + pad_size, projected_box[1] + pad_size), font, 0.5,(255,255,255), 1, cv2.LINE_AA)
    # img = project3Dto2DDraw(bbox3D[3:6], bbox3D[0:3], bbox3D[6], img, intrinsic)
    
cv2.imshow("p", img)
cv2.waitKey()