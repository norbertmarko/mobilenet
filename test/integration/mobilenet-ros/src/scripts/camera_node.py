#!/usr/bin/env python

import sys
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

def webcam_pub():
    pub = rospy.Publisher('webcam/image_raw', Image, queue_size=1)
    rospy.init_node('webcam_pub', anonymous=True)
    rate = rospy.Rate(1) # 60hz

    cam = cv2.VideoCapture(-1)
    bridge = CvBridge()

    if not cam.isOpened():
         sys.stdout.write("Webcam is not available")
         return -1

    while not rospy.is_shutdown():
        ret, frame = cam.read()
        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        #if ret:
        #   rospy.loginfo("Capturing image failed)

        pub.publish(msg)
        rospy.loginfo("Image Published")
        rate.sleep()


if __name__ == '__main__':
    try:
        webcam_pub()
    except rospy.ROSInterruptException:
        pass