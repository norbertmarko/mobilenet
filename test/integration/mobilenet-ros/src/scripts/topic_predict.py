#!/usr/bin/env python

#encoding='utf-8'

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# ROS related imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import conf

import os
import time

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.python.platform import gfile

# multithreading with ThreadPoolExecutor
import concurrent.futures

# ring buffer class
from utils.ringbuffer import RingBuffer

# color map for segmentation
colors = np.array(conf.colors_rgb)

# ROS integration part

def preprocess(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame_rgb, (conf.width, conf.height))


def callback(msg, storage):
    bridge = CvBridge()
    decodedFrame = bridge.imgmsg_to_cv2(msg)
    decodedFrame = preprocess(decodedFrame)
    storage.append(decodedFrame)

# producer
def listener(storage):
    rospy.init_node('webcam_listener', anonymous=True)
    
    # this is needed to pass multiple arguments to the callback function
    callback_lambda = lambda x: callback(x, storage)

    rospy.Subscriber(conf.topic, Image, callback_lambda)
    rospy.spin()

# prediction part

def consumer(storage):
    
    # leave time for storage to fill up
    time.sleep(5)

    # initialize the model
    model = tf.Graph()

    with model.as_default():
        f = gfile.FastGFile(conf.trt_opt_model, 'rb')
    
        graph_def = tf.GraphDef()
        # Parses a serialized binary message into the current message.
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

    sess = tf.Session(graph=model)

    softmax_tensor = sess.graph.get_tensor_by_name(conf.softmaxTensor)

    # variables for FPS logging
    log_fps = True
    rounds = 0
    round_start = None

    while True:

        if log_fps:
            if rounds == 1:
                round_start = time.perf_counter()
            
            if rounds == 101:
                round_finish = time.perf_counter()
                print('Rate: {} Hz'.format(100 / (round(round_finish-round_start, 2))))
                rounds = 0

        prediction = sess.run(softmax_tensor, {conf.inputTensor: storage.get().pop().reshape(-1, conf.height, conf.width, 3)})

        mask = np.argmax(prediction, axis=3)
        colored_mask = np.uint8(np.squeeze(colors[mask], axis=0))

        cv2.imshow('CameraFeed', colored_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sess.close()
            break

        rounds += 1

if __name__ == '__main__':
    
    #TODO: Test out different buffer sizes.
    storage = RingBuffer(20)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        t1 = executor.submit(listener, storage)
        t2 = executor.submit(consumer, storage)