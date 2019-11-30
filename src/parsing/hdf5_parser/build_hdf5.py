import sys
import numpy as np
import json
import progressbar
import cv2
import imutils

sys.path.append('../../..')

import config

from parsing.hdf5_parser.data_to_hdf5 import HDF5writer

def createHDF5(output_path, img_path, label_path, img_dims, label_dims,
                batchSize, width, height, classes, calcMean=True):

    # open HDF5writer
    writer = HDF5writer(img_dims, label_dims, output_path)
    
    # initialize lists to store MEAN values
    (R, G, B) = ([], [], [])

    assert len(img_path) == len(label_path), ("The number of images and the"
                                          "number of labels are not equal.")
    
    # initialize progressbar
    widgets = ["Converting data into HDF5:", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(img_path), widgets=widgets).start()

    for i in np.arange(0, len(img_path), batchSize):
        batchImages = img_path[i:i + batchSize]
        batchLabels = label_path[i:i + batchSize]

        # lists to store the batches during loop
        batch_img = []
        batch_label = []

        for (j, imagePath) in enumerate(batchImages):
            img = cv2.imread(imagePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img, (width, height))

            # mean claculation for actual image
            (r, g, b) = cv2.mean(img_array)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
            
            img_array.reshape(-1, height, width, 3)

            batch_img.append(img_array)

        for (k, labelPath) in enumerate(batchLabels):
            label = cv2.imread(labelPath)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            label = cv2.resize(label, (width, height))
            
            int_array = np.ndarray(shape=(height, width), dtype=int)
            int_array[:,:] = 0

            # rgb to integer
            for rgb, idx in classes.items():
                int_array[(label==rgb).all(2)] = idx

            int_array.reshape(-1, height, width, 1)

            batch_label.append(int_array)

        batch_img = np.array(batch_img).reshape(-1, height*width*3)
        batch_label = np.array(batch_label).reshape(-1, height*width*1)

        writer.add(batch_img, batch_label)
        
        # update ProgressBar
        pbar.update(i)

    pbar.finish()
    writer.close()
    #sys.stdout.flush()

    print("[INFO] dataset serialized successfully!")
    
    if calcMean:
        print("[INFO] serializing means...")

        D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
        f = open(config.dataset_mean, "w")
        f.write(json.dumps(D))
        f.close()

        print("[INFO] RGB mean values are calculated and"
               "saved across the dataset.")
