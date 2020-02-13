#!/usr/bin/env python

#encoding='utf-8'
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.platform import gfile
# ROS related imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import conf

# color map for segmentation
colors = np.array(conf.colors_rgb)

model_path = '/media/orion/6400F60300F5DC4C/nn_experiment/camera_test/mobilenet-master/optimization/export/trt_savedmodel/freezed_model_trt.pb'
inputTensor = 'input_1:0'
softmaxTensor = 'conv2d_transpose_2/truediv:0'

def init_tfmodel():
    trained_model = tf.Graph()
    with trained_model.as_default():
        f = gfile.FastGFile(conf.trt_opt_model,'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
    
    return trained_model

class TFROS():
    def __init__(self):
        trained_model = init_tfmodel()
        self._session = tf.Session(graph=trained_model)
        self._cv_bridge = CvBridge()
        
        self._sub = rospy.Subscriber(conf.topic, Image, 
                                       self.callback, queue_size=1)

    def preprocess(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame_rgb, (conf.width, conf.height))

    def callback(self, msg):
        decodedFrame = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        decodedFrame = self.preprocess(decodedFrame)

        softmax_tensor = self._session.graph.get_tensor_by_name(conf.softmaxTensor)
        prediction = self._session.run(
            softmax_tensor, {conf.inputTensor: decodedFrame.reshape(-1, conf.height, conf.width, 3)})
        
        mask = np.argmax(prediction, axis=3)
        colored_mask = np.uint8(np.squeeze(colors[mask], axis=0))
        
        # Publish as ROS topic
        seg_msg = self._cv_bridge.cv2_to_imgmsg(colored_mask, encoding="bgr8")
        pub = rospy.Publisher('tfros_seg', Image, queue_size=1)
        pub.publish(seg_msg)
   
        cv2.imshow('Prediction', colored_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self._session.close()
            

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('tf_ros')
    tfros = TFROS()
    tfros.main()
