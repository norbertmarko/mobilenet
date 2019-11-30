import sys
import tensorflow as tf
 
from tensorflow.keras.layers import Input

sys.path.append('../..')
import config

from skipnet import skipNet

IMG_SIZE = (config.height, config.width)
IMG_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)

num_classes = len(config.CLASSES)

ShowGraph = False
ShowLayerNum = False

''' ENCODER - needs import '''
# Feature Extractor without the classif. layer (pre-trained on ImageNet)
Encoder = tf.keras.applications.MobileNet(input_tensor=Input(shape=IMG_SHAPE),
                                            include_top=False,
                                            weights='imagenet')


# Encoder layers that connect into the Decoder
conv4_2_output = Encoder.get_layer(index=43).output
conv3_2_output = Encoder.get_layer(index=30).output 
conv_score_output = Encoder.output

''' DECODER - needs import '''
Decoder = skipNet(conv_score_output, conv4_2_output, conv3_2_output, num_classes)


if __name__ == "__main__":

    if ShowGraph:
        # import TensorFlow backend to help visualize graph
        from tensorflow.python.keras import backend as K
        from config import logdir
        
        graph = K.get_session().graph
        writer = tf.summary.FileWriter(logdir=logdir, graph=graph)
    

    if ShowLayerNum:
        for (i, layer) in enumerate(Encoder.layers):
            print("[INFO] {}\t{}".format(i, layer.__class__.__name__))
    





