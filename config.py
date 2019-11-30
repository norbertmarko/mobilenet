import os

# Path to images
img_dir = "./src/dataset/training/images"
# Path to labels
label_dir = "./src/dataset/training/labels"
# Option 1 - Path to validation data.
validation_path = './src/dataset/validation'

# Option 2 - if validation_path is not valid
dataset_length = len(os.listdir(img_dir))
#dataset_length = 2975
validation_split = 0.1
# Validation split ratio - only used if no validation_path exists.
val_size = int( dataset_length * validation_split )
#val_size = 500

# TFRECORD
# tfrecord data root dir
record_path = "./src/data/records/"
# training data output path
output_path_train = './src/data/records/train/training.tfrecord'
# validation data output path 
output_path_val = './src/data/records/val/validation.tfrecord'

# mean values json
dataset_mean = 'src/data/h5files/mean_values.json'

#HDF5
# training data output path
hdf5_path ='src/data/h5files/train_data.hdf5'
# validation data output path
hdf5_val_path ='src/data/h5files/val_data.hdf5'
# Serialization
bufferSize = None
hdf5BatchSize=32

# TensorBoard root logdir
logdir = "training/logs"

# input
width = 512
height = 288

# training
shuffle_buffer_size = 30
from params import batch_size
from params import val_batch_size

gpu_count = 1
LOG_ONLINE = True
show_summary = True
one_hot = True

# generator
steps_per_epoch = int( dataset_length // batch_size )
val_steps = int(val_size // val_batch_size)

# model paths - rewrite h5 path with desired model to be optimized
h5_model_dir = 'training/models'

saved_model_dir = 'export/savedmodel'
saved_model_opt_dir = 'export/opt_savedmodel'
saved_model_trt_dir = 'export/trt_savedmodel'

freezed_raw = os.path.join(saved_model_dir,'freezed_model_raw.pb')
freezed_opt = os.path.join(saved_model_opt_dir,'freezed_model_opt.pb')
freezed_trt = os.path.join(saved_model_trt_dir, "freezed_model_trt.pb")

# checkpoint weights
cps_dir = '/training/checkpoints/weights-improvement_stohastic_augment-{epoch:02d}-{val_acc:.2f}.hdf5'

# color maps for number of classes and mask coloring

CLASSES = {(1,64,128) : 0,
           (3,143,255) : 1, 
           (2,255,128) : 2,
           (255,140,0) : 3,
           (0,0,0):4}

import numpy as np
colors_rgb = np.array([[128, 64,1],
                       [255,143,3],
                       [128,255,2],
                       [0,140,255],
                       [0,  0,  0]])

colors_bgr = np.array([[1, 64,128],
                       [3,143,255],
                       [2,255,128],
                       [255,140,0],
                       [0,  0,  0]])