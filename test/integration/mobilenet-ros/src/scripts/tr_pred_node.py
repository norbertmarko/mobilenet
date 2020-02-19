#!/usr/bin/env python

#encoding='utf-8'
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class TRPRED():
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber('tfros_seg', Image,
                                        self.callback, queue_size=1)
    
    def callback(self, msg):
        decodedFrame = self._cv_bridge.imgmsg_to_cv2(msg, 'brg8')
        

        mrk_arr_msg = "" # should publish a marker array rosmsg

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('tr_pred')
    trpred = TRPRED()
    trpred.main()