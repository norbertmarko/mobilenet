import os

trt_opt_model = '/media/orion/6400F60300F5DC4C/nn_experiment/camera_test/mobilenet-master/optimization/export/trt_savedmodel/freezed_model_trt.pb'

height = 288 
width = 512

# TF config

inputTensor = 'input_1:0'
softmaxTensor = 'conv2d_transpose_2/truediv:0'

# ROS config

topic = 'webcam/image_raw'

# color maps for mask coloring

CLASSES = {(1,64,128) : 0,
           (3,143,255) : 1, 
           (2,255,128) : 2,
           (255,140,0) : 3,
           (0, 0, 0) :   4}

colors_rgb = [[128, 64,1],
              [255,143,3],
              [128,255,2],
              [0,140,255],
              [0,  0,  0]]

colors_bgr = [[1, 64,128],
              [3,143,255],
              [2,255,128],
              [255,140,0],
              [0,  0,  0]]

