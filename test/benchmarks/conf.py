import os

os.chdir('../..')
trt_opt_model = 'optimizer/export/trt_savedmodel/freezed_model_trt.pb'

# os.chdir changes path, give image path relative to that
single_image_path = './test/benchmarks/um_000000.png'
image_batch_path = './test/benchmarks/test_set'

height = 288 
width = 512

inputTensor = 'input_1:0'
softmaxTensor = 'conv2d_transpose_2/truediv:0'

# color maps for number of classes and mask coloring

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


