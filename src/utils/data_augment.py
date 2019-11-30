import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage

import numpy as np
import itertools
import sys

sys.path.append('../..')
import params

#parameters
# parameterize seed value

ia.seed(21)

seq = [None]

def load_aug():

    seq[0] = params.augmentation_list

'''
seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])
'''

def _augment_seg(img, seg):
    if seq[0] is None:
        load_aug()

# deterministic augmentation
    aug_sto = seq[0]

    images = []
    labels = []

    image_aug = aug_sto.augment_image( img )

    seg_aug = SegmentationMapOnImage(seg , nb_classes=np.max(seg)+1 , shape=img.shape)
    segmap_aug = aug_sto.augment_segmentation_maps( seg_aug )
    segmap_aug = segmap_aug.get_arr_int(background_class_id=3)

    return image_aug, segmap_aug

# image + label can be augmented in one line also
# image_aug, segmap_aug = aug_det(image=img, segmentation_maps=segmap)

def try_n_times(fn ,n ,*args ,**kwargs):

	attempts = 0

	while attempts < n:
		try:
			return fn(*args , **kwargs)
		except Exception as e:
			attempts += 1

	return fn(*args , **kwargs)
