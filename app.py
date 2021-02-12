
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import cv2
from flask import Flask, render_template,Response

sys.path.append("..")

from object_detection.utils import visualization_utils as vis_util, label_map_util

# # Model preparation
#y What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)



app = Flask(__name__)

# video_capture = cv2.VideoCapture(0)
frame_width = 680
frame_height = 480
fps = 30.0

video_capture = cv2.VideoCapture("tf_cam.mov")
def gen():

    while True:
        ret, image = video_capture.read()
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                ret, image_np = video_capture.read()
                image_np = np.array(image_np)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                """"Write below function in util from shredder machine"""
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2)

                for i, b in enumerate(boxes[0]):
                    #                 car                    bus                  truck
                    if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                        if scores[0][i] >= 0.5:
                            mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                            mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 2)
                            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x * 800), int(mid_y * 450)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                            if apx_distance <= 0.8:
                                if mid_x > 0 and mid_x < 0.5:
                                    cv2.putText(image_np, 'Safe Distance Alert!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                1.0, (255, 0, 0), 3)


                image= cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite('t.jpg', image)
        yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
    video_capture.release()


@app.route('/')
def index():
    """Video streaming"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()