import sys
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, add, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2

sys.path.append('../..')
import params

def skipNet(encoder_output, feed1, feed2, classes):

    # random initializer and regularizer
    init = params.weights_init
    reg = params.regularizer

    score_feed2 = Conv2D(kernel_size=(1, 1), filters=classes, padding="SAME",
                kernel_initializer=init, kernel_regularizer=reg)(feed2)
    score_feed2_bn = BatchNormalization()(score_feed2)
    score_feed1 = Conv2D(kernel_size=(1, 1), filters=classes, padding="SAME",
                kernel_initializer=init, kernel_regularizer=reg)(feed1)
    score_feed1_bn = BatchNormalization()(score_feed1)


    upscore2 = Conv2DTranspose(kernel_size=(4, 4), filters=classes, strides=(2, 2),
                               padding="SAME", kernel_initializer=init,
                               kernel_regularizer=reg)(encoder_output)
    upscore2_bn = BatchNormalization()(upscore2)

    fuse_feed1 = add([score_feed1_bn, upscore2_bn])

    upscore4 = Conv2DTranspose(kernel_size=(4, 4), filters=classes, strides=(2, 2),
                               padding="SAME", kernel_initializer=init,
                               kernel_regularizer=reg)(fuse_feed1)
    upscore4_bn = BatchNormalization()(upscore4)

    fuse_feed2 = add([score_feed2_bn, upscore4_bn])

    upscore8 = Conv2DTranspose(kernel_size=(16, 16), filters=classes, strides=(8, 8),
                               padding="SAME", kernel_initializer=init,
                               kernel_regularizer=reg, activation="softmax")(fuse_feed2)

    return upscore8
