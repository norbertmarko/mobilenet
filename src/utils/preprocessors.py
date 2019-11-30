import cv2
import h5py
import numpy as np

from utils.data_augment import try_n_times, _augment_seg

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):

        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):

        # splitting the image, subtracting channel-wise
        (B, G, R) = cv2.split(image.astype("float32"))

        # channel-wise subtraction
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        return cv2.merge([B, G, R])

class Normalize:
    def preprocess(self, image):
        image = np.float32(image) * (1. / 255)
        
        return image

class HDF5_generator:
    def __init__ (self, dbPath, image_shape, num_classes, batch_size, one_hot=True, preprocessors=None, do_augment=False):
        #opening the HDF5 database
        self.db = h5py.File(dbPath)
        # length of dataset
        self.ImageCount = np.array(self.db["images"]).shape[0]

        self.image_shape = image_shape
        self.width = self.image_shape[1]
        self.height = self.image_shape[0]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.preprocessors = preprocessors
        self.do_augment = do_augment

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.ImageCount, self.batch_size):
                # read in images and labels from HDF5
                images = self.db["images"][i: i + self.batch_size]
                labels = self.db["labels"][i: i + self.batch_size]


                if self.preprocessors is not None:
                    image_batch = []

                    for i in range(0, len(images)):
                        img = np.uint8(images[i].reshape(self.height, self.width, 3))
                        for p in self.preprocessors:
                            img = p.preprocess(img)
                        image_batch.append(img)

                    images = np.array(image_batch)
                else:
                    images = images.reshape(-1, self.height, self.width, 3)

                # check if data augmentation should be done
                if self.do_augment:

                    aug_images = []
                    aug_labels = []

                    for i in range(0, len(images)):
                        img = np.uint8(images[i].reshape(self.height, self.width, 3))
                        seg = np.uint8(labels[i].reshape(self.height, self.width, 1))

                        image, label = try_n_times(_augment_seg, 10, img, seg)
                        # appending the augmented images/labels to the batch
                        # batch size becomes bigger than the parameter - could be a problem
                        aug_images.append(image)
                        aug_labels.append(label)
                    # replacing them this way, so batch size stays the same (online augmentation)
                    images = np.array(aug_images)
                    labels = np.array(aug_labels)


                if self.one_hot:
                    label_batch = []
                    for i in range(0, len(images)):
                        label = np.uint8(labels[i].reshape(self.height, self.width))
                        one_hot_array = np.zeros((self.height, self.width, self.num_classes))

                        for c in range(self.num_classes):
                            one_hot_array[:, :, c] = (label == c).astype(int)

                        label_batch.append(one_hot_array)
                    labels = np.array(label_batch)

                yield (images, labels)
            epochs += 1

    def close(self):
        self.db.close()
