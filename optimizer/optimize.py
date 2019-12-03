#encoding='utf-8'

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import tensorflow as tf

import os
import numpy as np
from datetime import datetime
import argparse

sys.path.append('..')
import config

parser = argparse.ArgumentParser()
parser.add_argument("n", "--name", required=True, help="Name of the optimized model.")

TensorFlowOpt = True
TensorRtOpt = True

args = parser.parse_args()

h5_model = os.path.join('../src', config.h5_model_dir, args.n)

saved_model_dir = config.saved_model_dir
saved_model_opt_dir = config.saved_model_opt_dir
saved_model_trt_dir = config.saved_model_trt_dir

freezed_raw = config.freezed_raw
freezed_opt = config.freezed_opt
freezed_trt = config.freezed_trt


def get_graph_def_from_saved_model(saved_model_dir):
    
    print(saved_model_dir)
    print("")
    
    from tensorflow.python.saved_model import tag_constants
    
    with tf.Session() as session:
        meta_graph_def = tf.saved_model.loader.load(
            session,
            tags=[tag_constants.SERVING],
            export_dir=saved_model_dir
        )
        
    return meta_graph_def.graph_def


def get_graph_def_from_file(graph_filepath):
    
    print(graph_filepath)
    print("")
    
    from tensorflow.python import ops
    
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
            return graph_def


def describe_graph(graph_def, show_nodes=False):
    
    print(('Input Feature Nodes: {}'.format([node.name for node in graph_def.node if node.op=='Placeholder'])))
    print("")
    print ('Unused Nodes: {}'.format([node.name for node in graph_def.node if 'unused'  in node.name]))
    print("")
    print ('Output Nodes: {}'.format( [node.name for node in graph_def.node if 'softmax' in node.name]))
    print("")
    print ('Quanitization Nodes: {}'.format( [node.name for node in graph_def.node if 'quant' in node.name]))
    print("")
    print ('Constant Count: {}'.format( len([node for node in graph_def.node if node.op=='Const'])))
    print("")
    print ('Variable Count: {}'.format( len([node for node in graph_def.node if 'Variable' in node.op])))
    print("")
    print ('Identity Count: {}'.format( len([node for node in graph_def.node if node.op=='Identity'])))
    print("")
    print('Total nodes: {}'.format( len(graph_def.node)))
    print("")
    
    if show_nodes==True:
        for node in graph_def.node:
            print('Op:{} - Name: {}'.format(node.op, node.name))


def get_size(model_dir):
    
    print(model_dir)
    print("")
    
    pb_size = os.path.getsize(os.path.join(model_dir,'saved_model.pb'))
    
    variables_size = 0
    if os.path.exists(os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
        variables_size = os.path.getsize(os.path.join(model_dir,'variables/variables.data-00000-of-00001'))
        variables_size += os.path.getsize(os.path.join(model_dir,'variables/variables.index'))

    print("Model size: {} KB".format(round(pb_size/(1024.0),3)))
    print("Variables size: {} KB".format(round( variables_size/(1024.0),3)))
    print("Total Size: {} KB".format(round((pb_size + variables_size)/(1024.0),3)))


def freeze_graph(saved_model_dir, stage):
    
    assert stage in ['raw', 'opt', 'trt'], 'stage argument must be either raw, opt or trt'

    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.saved_model import tag_constants
    
    output_graph_filename = os.path.join(saved_model_dir, "freezed_model_{}.pb".format(stage))
    output_node_names = "conv2d_transpose_2/truediv"
    initializer_nodes = ""

    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags = tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,

        input_graph=None, 
        input_saver=False,
        input_binary=False, 
        input_checkpoint=None, 
        restore_op_name=None, 
        filename_tensor_name=None, 
        clear_devices=False,
        input_meta_graph=False,
    )
    
    print("SavedModel graph freezed!")


# GRAPH TRANSFORM TOOL
def optimize_graph(model_dir, graph_filename, transforms):
    
    from tensorflow.tools.graph_transforms import TransformGraph
    
    input_names = ['input_1']
    output_names = ['conv2d_transpose_2/truediv']
    
    graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
    optimized_graph_def = TransformGraph(graph_def, 
                                         input_names,
                                         output_names,
                                         transforms 
                                        )
    tf.train.write_graph(optimized_graph_def,
                        logdir=model_dir,
                        as_text=False,
                        name='freezed_model_opt.pb')
    
    print("Freezed graph optimized!")

# the list of optimizations used
transforms = [
    'remove_nodes(op=Identity)', 
    'fold_constants(ignore_errors=true)',
    'fold_batch_norms',
#    'fuse_resize_pad_and_conv',
#    'quantize_weights',
#    'quantize_nodes',
    'merge_duplicate_nodes',
    'strip_unused_nodes', 
    'sort_by_execution_order'
]


def convert_graph_def_to_saved_model(graph_filepath):

    from tensorflow.python import ops
    export_dir=os.path.join(saved_model_dir,'optimized')

    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)

    graph_def = get_graph_def_from_file(graph_filepath)
    
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name="")
        tf.saved_model.simple_save(session,
                export_dir,
                inputs={
                    node.name: session.graph.get_tensor_by_name("{}:0".format(node.name)) 
                    for node in graph_def.node if node.op=='Placeholder'},
                outputs={
                    "softmax": session.graph.get_tensor_by_name("conv2d_transpose_2/truediv:0"),
                }
            )

        print("Optimized graph converted to SavedModel!")

if __name__=='__main__':

    if TensorFlowOpt:
        # load the .h5 / .model
        model = tf.keras.models.load_model(h5_model)

        # export the SavedModel format
        tf.keras.experimental.export_saved_model(model, saved_model_dir)


        # TENSORFLOW OPTIMIZATION

        # get initial information from the graph def
        # display information
        describe_graph(get_graph_def_from_saved_model(saved_model_dir))

        # display size information
        get_size(saved_model_dir)

        # freezing the saved model graph
        freeze_graph(saved_model_dir, raw)

        # describe graph after freezing it
        describe_graph(get_graph_def_from_file(freezed_raw))

        # optimization function
        optimize_graph(saved_model_dir, 'freezed_model_raw.pb', transforms)

        # describe graph after optimization
        describe_graph(get_graph_def_from_file(freezed_opt))

        # convert frozen model back to SavedModel
        convert_graph_def_to_saved_model(freezed_opt)

    if TensorRtOpt:

        # TENSORRT OPTIMIZATION (savedModel variation)

        import tensorflow.contrib.tensorrt as trt
        from tensorflow.python.platform import gfile


        # optimizer uses these lists
        output_node_names = ['conv2d_transpose_2/truediv']
        outputs =  ['conv2d_transpose_2/truediv:0']

        tf.reset_default_graph()

        graph = tf.Graph()

        # optimization process
        with graph.as_default():
            with tf.Session() as sess:
                
                trt_graph = trt.create_inference_graph(
                    input_graph_def=None,
                    outputs=outputs,
                    input_saved_model_dir=saved_model_opt_dir,
                    input_saved_model_tags=['serve'],
                    max_batch_size=20,
                    max_workspace_size_bytes=7000000000,
                    precision_mode='INT8')

                for node in trt_graph.node:
                    if node.op=='Placeholder':
                        print("input {}".format(node))

                output_stuff = tf.import_graph_def(trt_graph, name="", return_elements=outputs)

                tf.saved_model.simple_save(sess, saved_model_trt_dir,
                    inputs={'input_image': graph.get_tensor_by_name('input_1:0')},
                    outputs={'result':graph.get_tensor_by_name('conv2d_transpose_2/truediv:0')}
                    )


        # after tensorrt opt is done, use these scripts to test file and predictions

        # describe graph after TensorRT optimization
        describe_graph(get_graph_def_from_saved_model(saved_model_trt_dir))

        # freezing the optimized saved model graph
        freeze_graph(saved_model_trt_dir, "trt")



