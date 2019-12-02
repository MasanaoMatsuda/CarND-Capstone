import os
import rospy
import cv2
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self, model_file):
        self.current_light = TrafficLight.UNKNOWN
        
        cwd = os.path.dirname(os.path.realpath(__file__))

        model_path = os.path.join(cwd, "train_model/{}".format(model_file))
        rospy.logwarn("model_path={}".format(model_path))

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.category_idx = {1: {'id': 1, 'name': 'Green'},
                             2: {'id': 2, 'name': 'Red'},
                             3: {'id': 3, 'name': 'Yellow'},
                             4: {'id': 4, 'name': 'off'}}
        

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # allow allocating gpu memory when it's needed.

        self.sess = tf.Session(graph=self.detection_graph, config=config)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width, height, _ = image_rgb.shape
        iamge_np = np.expand_dims(image_rgb, axis=0)

        with self.detection_graph.as_default():
            boxes, scores, classes, num = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        min_score_thresh = .5
        red_count = 0
        count = 0

        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                count += 1
                class_name = self.category_idx[classes[i]]['name']

                if class_name == 'Red':
                    red_count += 1

        self.current_light = TrafficLight.GREEN if red_count < count - red_count else TrafficLight.RED
        
        return self.current_light
