#encoding='utf-8'

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

sys.path.append('..')

import config
from parsing.path_list_generator import PathListGenerator
from parsing.hdf5_parser.data_to_hdf5 import HDF5writer
from parsing.hdf5_parser.build_hdf5 import createHDF5

#TODO: make TFrecord selectable

# import paths
path = PathListGenerator()

path.img_dir = config.img_dir
path.label_dir = config.label_dir
path.val_size = config.val_size
path.validation_path = config.validation_path

if __name__=='__main__':

    # unpack usable paths
    (train_img, train_lbl, val_img, val_lbl) = path.build()

    # calculate input dimensions
    height = config.height
    width = config.width

    classes = config.CLASSES

    img_dims = (len(train_img), height*width*3)
    label_dims = (len(train_lbl), height*width*1)

    v_img_dims = (len(val_img), height*width*3)
    v_label_dims = (len(val_lbl), height*width*1)

    hdf5_path = config.hdf5_path
    hdf5_val_path = config.hdf5_val_path

    hdf5BatchSize = config.hdf5BatchSize

    # create training data
    createHDF5(hdf5_path, train_img, train_lbl, img_dims, label_dims,
            hdf5BatchSize, width, height, classes)
    # create validation data 
    createHDF5(hdf5_val_path, val_img, val_lbl, v_img_dims, v_label_dims,
            hdf5BatchSize, width, height, classes, calcMean=False)