#encoding='utf-8'

import sys
import os
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import tensorflow as tf
from tensorflow.python.platform import gfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import conf

colors = np.array(conf.colors_bgr)

IMG_SHAPE = (conf.height, conf.width, 3)

image_path = conf.single_image_path

def prepare(path):
    img = cv2.imread(path)
    b,g,r = cv2.split(img)
    img_array_rgb = cv2.merge([r,g,b])
    img_array = cv2.resize(img_array_rgb, (IMG_SHAPE[1], IMG_SHAPE[0]))

    return img_array.reshape(-1, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])

# initialize the model
model = tf.Graph()

# Load in frozen graph
with model.as_default():
    f = gfile.FastGFile(conf.trt_opt_model, 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

test_image = prepare(image_path)

# Prediction
with model.as_default():
    with tf.Session(graph=model) as sess:

        softmax_tensor = sess.graph.get_tensor_by_name(conf.softmaxTensor)
        prediction = sess.run(softmax_tensor, {conf.inputTensor: test_image})
        
        mask = np.argmax(prediction[0], axis=2)

        colored_mask = colors[mask]

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.imshow(test_image.reshape(conf.height, conf.width, 3))
        ax2.imshow(colored_mask)
        plt.show()



        


        
    
    