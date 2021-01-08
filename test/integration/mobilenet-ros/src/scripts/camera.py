#!/usr/bin/env python

#encoding='utf-8'

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def camera_stream_publisher(device_number=0):

    '''
    Publishes the video frames from the device given
    as input. For more info, look up cv2.VideoCapture()
    function. 

    device_number: The number of the input device (default = -1). 
    '''

    publisher = rospy.Publisher('camera_stream/raw', Image, queue_size=1)
    rospy.init_node('camera_stream', anonymous=True)
    # publishing rate frequency (Hz)
    rate = rospy.Rate(60) 

    camera = cv2.VideoCapture(device_number)
    bridge = CvBridge()

    if not camera.isOpened():
         sys.stdout.write("No camera detected.")
         return -1

    while not rospy.is_shutdown():
        _ , frame = camera.read()
        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        publisher.publish(msg)
        rospy.loginfo("[INFO] Frame Published.")
        rate.sleep()


if __name__ == '__main__':
    try:
        camera_stream_publisher()
    except rospy.ROSInterruptException:
        pass