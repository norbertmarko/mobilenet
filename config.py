import os

# Path to images
img_dir = "./dataset/training/images"
# Path to labels
label_dir = "./dataset/training/labels"
# Option 1 - Path to validation data.
validation_path = './dataset/validation'

# Option 2 - if validation_path is not valid
#dataset_length = len(os.listdir(img_dir))
dataset_length = 2975
validation_split = 0.1
# Validation split ratio - only used if no validation_path exists.
#val_size = int( dataset_length * validation_split )
val_size = 500

# TFRECORD
# tfrecord data root dir
record_path = "./records/"
# training data output path
output_path_train = './records/train/training.tfrecord'
# validation data output path 
output_path_val = './records/val/validation.tfrecord'

# mean values json
dataset_mean = 'h5files/mean_values.json'

#HDF5
# training data output path
hdf5_path ='h5files/train_data.hdf5'
# validation data output path
hdf5_val_path ='h5files/val_data.hdf5'
# Serialization
bufferSize = None
hdf5BatchSize=32

# TensorBoard root logdir
logdir = "./logs"

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
h5_model = os.path.join('training', 'saved_models', 'mobilenet-v1-skipnet.model')

saved_model_dir = './export/savedmodel'
saved_model_opt_dir = './export/opt_savedmodel'
saved_model_trt_dir = './export/trt_savedmodel'

freezed_raw = os.path.join(saved_model_dir,'freezed_model_raw.pb')
freezed_opt = os.path.join(saved_model_opt_dir,'freezed_model_opt.pb')
freezed_trt = os.path.join(saved_model_trt_dir, "freezed_model_trt.pb")

#TODO: take this out after done
trt_opt_model = '/media/orion/6400F60300F5DC4C/nn_experiment/camera_test/mobilenet-master/optimization/export/trt_savedmodel/freezed_model_trt.pb'

# checkpoint weights
cps_dir = './cps/weights-improvement_stohastic_augment-{epoch:02d}-{val_acc:.2f}.hdf5'

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