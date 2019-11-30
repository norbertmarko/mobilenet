# unopt graph
# INPUT: input_1 (input_1:0)
# OUTPUT: conv2d_transpose_2/truediv (conv2d_transpose_2/truediv:0)

#opt graph
# INPUT: input_1 (input_1:0)
# OUTPUT: conv2d_transpose_2/truediv (conv2d_transpose_2/truediv:0)

import tensorflow as tf
import sys

sys.path.append('..')
import config

# frozen graph paths

unopt_graph = config.freezed_raw
opt_graph = config.freezed_opt

logdir = config.logdir

# read in graphDef
from tensorflow.python.platform import gfile

chosenGraph = opt_graph # enter the chosen path variable here 

model = tf.Graph()

with model.as_default():
    f = gfile.FastGFile(chosenGraph, 'rb')
    graph_def = tf.GraphDef()

    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# PRINT I/O nodes

# helper

from tensorflow.python import ops

def get_graph_def_from_file(graph_filepath):

    tf.reset_default_graph()

    with model.as_default():
        f = gfile.FastGFile(graph_filepath, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        return graph_def

# unoptimized version
unopt_graph_def = get_graph_def_from_file(unopt_graph)

#tf.import_graph_def(unopt_graph_def, name="")

for node in unopt_graph_def.node:
    if node.op=='Placeholder':
        print("unopt input {}".format(node)) # this will be the input node

# optimized version
opt_graph_def = get_graph_def_from_file(opt_graph)

#tf.import_graph_def(opt_graph_def, name="")

for node in opt_graph_def.node:
    if node.op=='Placeholder':
        print("opt input {}".format(node)) # this will be the input node






