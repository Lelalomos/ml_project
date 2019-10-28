#convert size img = 1280,720
# Import packages
from PIL import Image
from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'IMG_1782.JPG'
outputFile = 'pre_'+IMAGE_NAME
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 5

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_dpi = Image.open(PATH_TO_IMAGE)

image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')

counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1, image,
                                                                                             1,
                                                                                             0,
                                                                                             np.squeeze(
                                                                                                 boxes),
                                                                                             np.squeeze(classes).astype(
                                                                                                 np.int32),
                                                                                             np.squeeze(
                                                                                                 scores),
                                                                                             category_index,
                                                                                             use_normalized_coordinates=True,
                                                                                             min_score_thresh=0.3,
                                                                                             line_thickness=8)
boxs = np.squeeze(boxes)
score = np.squeeze(scores)
pre_class = counting_mode.split(':')
# pre_class = pre_class.replace("'","")
font = cv2.FONT_HERSHEY_SIMPLEX
if(len(counting_mode) == 0):
    cv2.putText(image, "...", (10, 35), font, 0.8,
                (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
else:
    cv2.putText(image, counting_mode, (10, 35), font, 0.8,
                (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)

    # box = []

# convert JPG to PNG
def convert_pixel2inch(x2, y2, images=image_dpi):
    img_split = IMAGE_NAME.split('.', 1)
    if img_split[-1] == 'png' or img_split[-1] == 'PNG':
        img = Image.open(PATH_TO_IMAGE)
        dpi = img.info['dpi']
        xmax_inch = round(x2/dpi[0], 2)
        ymax_inch = round(y2/dpi[0], 2)
    else:
        img = Image.open(PATH_TO_IMAGE)
        img_files = img_split[0]+'.png'
        PATH_2_IMAGE = os.path.join(CWD_PATH, img_files)
        img.save(PATH_2_IMAGE, quality=95, dpi=(96, 96))

        img_convert = Image.open(PATH_2_IMAGE)
        dpi = img_convert.info['dpi']
        xmax_inch = round(x2/96, 2)
        ymax_inch = round(y2/96, 2)
    return xmax_inch, ymax_inch


height, width, d = image.shape
for i in range(0, len(score)):
        # print(boxes[[0], [i]])
    if(score[i] >= 0.3):
        ymin, xmin, ymax, xmax = boxs[i]
        x1 = int(xmin*width)
        x2 = int(xmax*width)
        y1 = int(ymin*height)
        y2 = int(ymax*height)

        xmax, ymax = convert_pixel2inch(x2, y2)

        cv2.putText(image, 'w: {} , h: {}'.format(xmax, ymax),
                    (x1, y1), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        print('w: {} , h: {}'.format(xmax, ymax))


cv2.imwrite(outputFile, image.astype(np.uint8))
print('suss')

# All the results have been drawn on image. Now display the image.
# cv2.imshow('Object detector', image)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
