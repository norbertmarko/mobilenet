#!/usr/bin/env python

#encoding='utf-8'

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import time

import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import conf

# set cuda device number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# better alternative to: "config.gpu_options.allow_growth = True"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# color map for segmentation
colors = np.array(conf.colors_rgb_cs)


def init_tf_model():
    trained_model = tf.Graph()
    with trained_model.as_default():
        f = gfile.FastGFile(
            conf.trt_opt_model,'rb'
        )
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
    return trained_model


class TensorFlowSegmentationROSNode():
    '''
    docstring
    '''
    def __init__(self, gpu_memory_fraction=0.005, show_stream=True):
        self.TrainedModel = init_tf_model()

        self.GPU_OPTIONS = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction
        )
        self.TFConfig = tf.ConfigProto(gpu_options=self.GPU_OPTIONS)
        self.Session = tf.Session(graph=self.TrainedModel, config=self.TFConfig)

        self.CvBridge = CvBridge()
        self.Subscriber = rospy.Subscriber(
            conf.topic, Image, self.callback, queue_size=1
        )

        self.show_stream = show_stream
        self.pred_time_list = []

        # self.COLORS, switch to say RGB / BGR
    
    def preprocess(self, frame):
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, (conf.width, conf.height))
        
        #reshape itt!!!

        # only publish the important part, so no delay, preprocess easier! switch , data structure!!
        # start time, keep measuring, only stop clock at the end of the average, divide by number
        # of preds

    def callback(self, ros_msg):
        frame = self.CvBridge.imgmsg_to_cv2(ros_msg, "bgr8")
        frame = self.preprocess(frame)
        softmax_tensor = self.Session.graph.get_tensor_by_name(conf.softmaxTensor)

        start = time.clock()
        prediction = self.Session.run(
            softmax_tensor, {conf.inputTensor: frame.reshape(-1, conf.height, conf.width, 3)}
        )
        end = time.clock()

        self.pred_time_list.append(end-start)
        
        if len(self.pred_time_list) % 10 == 0:
            print("[INFO] Mean FPS: {}".format(
                1/(sum(self.pred_time_list)/len(self.pred_time_list))
            ))
            print("[INFO] Single Prediction Time: {}".format(
                end - start
            ))
            self.pred_time_list = []

        mask = np.argmax(prediction, axis=3)
        mask = np.uint8(np.squeeze(colors[mask], axis=0))

        # Publish as ROS topic
        ros_msg_seg = self.CvBridge.cv2_to_imgmsg(mask, encoding="bgr8")
        publisher = rospy.Publisher('camera_stream/segmentation', Image, queue_size=1)
        publisher.publish(ros_msg_seg)

        #Only needed to check video stream on PC
        if self.show_stream:
            cv2.imshow('camera_stream/segmentation', mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.Session.close()

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('segmentation_prediction')
    tf_seg_ros = TensorFlowSegmentationROSNode()
    tf_seg_ros.main()
