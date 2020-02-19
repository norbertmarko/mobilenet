#!/usr/bin/env python

#encoding='utf-8'
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge, CvBridgeError

from conf import height as h
from conf import width as w

class TransformPrediction():

    def __init__(self):
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber('tfros_seg', Image,
                                        self.callback, queue_size=1)

    def _birds_eye_transform(self, frame, source, destination):
        # Transformation Matrix
        tr_matrix = cv2.getPerspectiveTransform(source, destination)
        return cv2.warpPerspective(frame, tr_matrix, (w, h))
    
    def callback(self, msg):
        decodedFrame = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Source Points - point pairs - (top left, top right, bottom left, bottom right)
        sourcePoints = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])

        # Destination Points
        destinationPoints = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])

        warpedFrame = self._birds_eye_transform(decodedFrame,
                                                sourcePoints, destinationPoints)

        # extract color information - road
        road_arr = warpedFrame[0:140:255]

        # Extract Markers

        mrk_arr_msg = "" # should publish a MarkerArray rosmsg
        
        # Publish MarkerArray
        pub = rospy.Publisher('markerarray_seg', MarkerArray, queue_size=1)
        pub.publish(mrk_arr_msg)

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('tr_pred')
    trpred = TransformPrediction()
    trpred.main()