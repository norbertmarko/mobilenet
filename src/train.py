#encoding='utf-8'

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

sys.path.append('..')

import params
import config

import json
import os
import time

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
from training.model_parts import Encoder, Decoder

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from utils.preprocessors import HDF5_generator, Normalize, MeanPreprocessor

import wandb
from wandb.keras import WandbCallback
wandb.init(project=params.project_name, name=params.run_name)

if not config.LOG_ONLINE:
    os.environ['WANDB_MODE'] = 'dryrun'

# logged hyperparameters
wandb.config.weight_decay = params.weight_decay

wandb.config.epochs_wu = params.epochs_wu
wandb.config.epochs_ft = params.epochs_ft

wandb.config.batch_size = params.batch_size
wandb.config.val_batch_size = params.val_batch_size

wandb.config.learning_rate_warmup = params.learning_rate_wu 
wandb.config.warmupStartLayer = params.warmupStartLayer
wandb.config.warmup_opt = params.warmupOptimizer

wandb.config.learning_rate_finetune = params.learning_rate_ft
wandb.config.finetuneStartLayer = params.finetuneStartLayer
wandb.config.finetune_opt = params.finetuneOptimizer

wandb.config.augmentation = params.augmentation


# define model from encoder and decoder
def mobileNet_v1_skipNet():
    return Model(inputs=Encoder.input, outputs=Decoder)


# CALLBACKS
tensorboard = TensorBoard(log_dir=os.path.join(config.logdir, params.tensorboard_name))
checkpoint = ModelCheckpoint(config.cps_dir, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')
wandbcallback = WandbCallback()

callbacks_list = [checkpoint, tensorboard, wandbcallback]

# PREPROCESSORS

# mean subtraction
means = json.loads(open(config.dataset_mean).read())
mp = MeanPreprocessor(means["R"], means["G"], means["B"])

# normalize
norm = Normalize()

preprocessor_list = [None]

# generator settings
hdf5_path = config.hdf5_path
hdf5_val_path = config.hdf5_val_path
image_shape = (config.height, config.width, 3)
wandb.config.num_classes = len(config.CLASSES)
one_hot = config.one_hot

# training data generator
train_gen = HDF5_generator(hdf5_path, image_shape,
                           wandb.config.num_classes, 
                           wandb.config.batch_size,
                           one_hot, 
                           preprocessors=None, 
                           do_augment=wandb.config.augmentation)

# validation data generator
val_gen = HDF5_generator(hdf5_val_path, image_shape,
                         wandb.config.num_classes, 
                         wandb.config.val_batch_size,
                         one_hot, 
                         preprocessors=None, 
                         do_augment=False)

# instantiate model
if config.gpu_count <= 1:
    print("[INFO] Training network with 1 GPU...")
    model = mobileNet_v1_skipNet()

else:
    print("[INFO] Training with {} GPUs...".format(config.gpu_count))
    with tf.device("/cpu:0"):
        model = mobileNet_v1_skipNet()
    # making model parallel
    model = multi_gpu_model(model, config.gpu_count)

# Print out hyperparams
from utils import helper
helper.printHyperparams() 

# TRAINING

for layer in Encoder.layers:
    layer.trainable = False

# WARM UP
for layer in Encoder.layers[params.warmupStartLayer:]:
    layer.trainable = True

print("[INFO] warming up model head...")
model.compile(optimizer=params.warmupOptimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

if config.show_summary:
    model.summary()

# backpropagation settings
steps_per_epoch = config.steps_per_epoch
val_steps = config.val_steps

model.fit_generator(train_gen.generator(), steps_per_epoch,
                                        wandb.config.epochs_wu,
                                        validation_data=val_gen.generator(), 
                                        validation_steps=val_steps,
                                        callbacks=callbacks_list)

# FINE TUNING
for layer in Encoder.layers[params.finetuneStartLayer:]:
    layer.trainable = True

print("[INFO] re-compiling model for fine-tuning...")
model.compile(optimizer=params.finetuneOptimizer, 
              loss='categorical_crossentropy', metrics=['accuracy'])


model.fit_generator(train_gen.generator(), steps_per_epoch, 
                                        wandb.config.epochs_ft,
                                        validation_data=val_gen.generator(), 
                                        validation_steps=val_steps,
                                        callbacks=callbacks_list)

# saving weights / model
model.save_weights(os.path.join(wandb.run.dir,"{}.h5".format(params.tensorboard_name)),
                                                                        overwrite=True)
model.save(os.path.join(config.h5_model_dir, "{}.model".format(params.tensorboard_name)), overwrite=True)
