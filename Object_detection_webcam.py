######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

# Some of the code is copied from Google's example at
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# and some is copied from Dat Tran's example at
# https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

# but I changed it to make it more understandable to me.


# Import packages
from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Import utilites
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 6

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

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3, 1280)
ret = video.set(4, 720)

acc = 0.6
while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(video.get(1),
                                                                                          frame,
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
                                                                                          min_score_thresh=0.9,
                                                                                          line_thickness=8)
    boxs = np.squeeze(boxes)
    score = np.squeeze(scores)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if(len(counting_mode) == 0):
        cv2.putText(frame, "...", (10, 35), font, 0.8,
                    (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
    else:
        cv2.putText(frame, counting_mode, (10, 35), font, 0.8,
                    (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)

    # box = []
    height, width, d = frame.shape
    for i in range(0, len(score)):
        # print(boxes[[0], [i]])
        if(score[i] >= 0.9):
            ymin, xmin, ymax, xmax = boxs[i]
            x1 = int(xmin*width)
            x2 = int(xmax*width)
            y1 = int(ymin*height)
            y2 = int(ymax*height)
            cv2.putText(frame, 'w: {} , h: {}'.format(x2,y2) ,(x1,y1),font, 1,(0,0,255),2,cv2.LINE_AA)
    # print('x1: {} , x2: {} , y1: {} , y2
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
