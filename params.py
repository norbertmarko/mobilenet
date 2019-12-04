import os
import datetime

from imgaug import augmenters as iaa

# hyperparam imports
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop

now = datetime.datetime.now()
wandb_name = "run-{}-{}-{:d}-{:02d}".format(now.month, 
                                         now.day, now.hour, now.minute)

tensorboard_name = "mobilenet-v1-skipnet-{}-{}-{:d}-{:02d}".format(now.month, 
                                               now.day, now.hour, now.minute)

# HYPERPARAMETERS

project_name = "mobilenet-v1-skipnet_288_512_5"
run_name = wandb_name

epochs_wu = 15
epochs_ft = 135

batch_size = 10
val_batch_size = 8

stddev = 0.01
weights_init = RandomNormal(stddev=stddev)

weight_decay = 1e-3 
regularizer = l2(weight_decay)

learning_rate_wu = 4e-3
warmupOptimizer = RMSprop(lr=learning_rate_wu)
warmupStartLayer = 87 #81

learning_rate_ft = 4e-4
finetuneOptimizer = Adam(lr=learning_rate_ft)
finetuneStartLayer = 0 #44

augmentation = True
augType = 'det' # sto / det

seedValue = 21

# list of augmentations

augmentation_list = iaa.SomeOf((0, None), [
        iaa.Dropout([0.05, 0.1]),      # drop 5% or 20% of all pixels
        #iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-15, 15), shear=(15)),  # rotate by -45 to 45 degrees (affects segmaps)
        iaa.Fliplr(0.5),
        #iaa.ElasticTransformation(alpha=25, sigma=5),  # apply water effect (affects segmaps) alpha was 50
        iaa.Multiply((0.5, 1.5)) # birghtness change
    ], random_order=True)
