import numpy as np
import os
import cv2
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

cap = cv2.VideoCapture('H:/Projects/AITrafficControlSystem/SampleVid/traffic_video.mp4')

# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'rfcn_resnet101_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = 'H:/Projects/AITrafficControlSystem/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
PATH_TO_LABELS = 'H:\Projects\AITrafficControlSystem\object_detection\data\mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# Download Model
#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#    file_name = os.path.basename(file.name)
#    if 'frozen_inference_graph.pb' in file_name:
#        tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.sizeq
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def lane_mask(lane_image):
    gray_lane_image = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
    # remove gaussian noise
    gaussian_lane_image = cv2.GaussianBlur(gray_lane_image, (3, 3), 0)
    # edge and line detection
    canny_lane_image = cv2.Canny(gaussian_lane_image, 50, 150)
    # we need to consider the region of interest , coordinates of a lane
    points = np.array([[280, 50], [150, 700], [450, 700], [325, 50]])

    # Cropping the bounding rectangle
    lane = cv2.boundingRect(points)
    x, y, w, h = lane
    croped_lane = lane_image[y:y + h, x:x + w]

    # making a mask
    points = points - points.min(axis=0)
    mask_lane = np.zeros(croped_lane.shape[:2], np.uint8)
    x = cv2.drawContours(mask_lane, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # bit-op
    result_lane1 = cv2.bitwise_and(croped_lane, croped_lane, mask=mask_lane)

    # add the white background
    background = np.ones_like(croped_lane, np.uint8) * 255
    cv2.bitwise_not(background, background, mask=mask_lane)
    result_lane2 = background + result_lane1

    return result_lane2

# Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # Getting required lane section as input to analyse
            image_np = lane_mask(image_np)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Display output
            cv2.imshow('object detection', image_np)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
