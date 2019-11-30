import os

os.chdir('../..')
trt_opt_model = 'optimizer/export/trt_savedmodel/freezed_model_trt.pb'

height = 288 
width = 512

colors = [[128, 64,1],
          [255,143,3],
          [128,255,2],
          [0,140,255],
          [0,  0,  0]]

