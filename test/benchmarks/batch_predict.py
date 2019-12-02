import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from datetime import datetime

import conf

from utils.ringbuffer import RingBuffer

def prepare(path):
    img = cv2.imread(path)
    b,g,r = cv2.split(img)
    img_array_rgb = cv2.merge([r,g,b])
    
    return cv2.resize(img_array_rgb, (512, 288))


colors = np.array(conf.colors_bgr)

# image paths
image_list = []
for path in range(len(os.listdir(conf.image_batch_path))):
    image_list.append(os.path.join(conf.image_batch_path, os.listdir(conf.image_batch_path)[path]))

storage = RingBuffer(20)

for i in range(20):
    storage.append(prepare(image_list[i]))

# initialize the model
model = tf.Graph()

with model.as_default():
    f = gfile.FastGFile(conf.trt_opt_model, 'rb')
    
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

# initialize a session in a way it stays open on the device - close it at the end

sess = tf.Session(graph=model)

softmax_tensor = sess.graph.get_tensor_by_name(conf.softmaxTensor)

batch_time = []

j = 0
while j < 20:
    k = cv2.waitKey(33)
    
    time_start = datetime.utcnow()
    prediction = sess.run(softmax_tensor, {conf.inputTensor: storage.get()[j].reshape(-1, conf.height, conf.width, 3)})
    time_end = datetime.utcnow()
    
    time_elapsed_sec = (time_end - time_start).total_seconds()
    
    batch_time.append(time_elapsed_sec)

    print("Image {} inference time: {}".format(j+1, time_elapsed_sec))
    
    j += 1

    if k==27:    
        sess.close()
        break

    if j == 19:
        mask = np.argmax(prediction, axis=3)

        colored_mask = colors[mask]

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.imshow(storage.get()[j])
        ax2.imshow(np.squeeze(colored_mask, axis=0))
        plt.show()

sess.close()

batch_time = batch_time[1:]

print("Average latency per batch: {} seconds".format(sum(batch_time)/(len(batch_time))))
print("Average prediction frequency: {} Hz".format(1/(sum(batch_time)/(len(batch_time)))))
