import random
import os

class PathListGenerator:
    """This class outputs lists of file paths to the training data
    and validation data separately. You can also split your data into
    a training and validation part, if this isn't implemented by the
    directory structure."""
    def __init__(self):
        # append to these before using the class
        self.img_dir = None
        self.label_dir = None

        self.val_size = None
        
        self.validation_path = None 

    def get_path(self):
        """Get paths to images and labels."""
        imagePaths = []
        labelPaths = []

        for image in sorted(os.listdir(self.img_dir)):
            imagePaths.append(image)
            
        for label in sorted(os.listdir(self.label_dir)):
            labelPaths.append(label)
            
        return (imagePaths, labelPaths)
    
    def shuffle(self, img_path, lbl_path):

        """Shuffles the data."""
        c = list(zip(img_path, lbl_path))
        random.shuffle(c)
        (img_path, lbl_path) = zip(*c)
        
        assert img_path == lbl_path, "Shuffle problem, paths are not in sync!"
        
        img_shuffled = []
        lbl_shuffled = []

        for img in img_path:
            img_shuffled.append(os.path.join(self.img_dir, img))
        for lbl in lbl_path:
            lbl_shuffled.append(os.path.join(self.label_dir, lbl))
        
        return img_shuffled, lbl_shuffled

    def val_isdir(self):
        """Checks if the configured validation directory exists."""
        return os.path.isdir(self.validation_path)

    def split_data(self, val_exist, imagePaths, labelPaths):
        """Splits training data if there isn't a separate validation set."""
        self.val_exist = val_exist

        if val_exist:
            trainPaths_img = imagePaths
            trainPaths_lbl = labelPaths

            valPaths_img = []
            valPaths_lbl = []

            for image in sorted(os.listdir(os.path.join(self.validation_path, "images"))):
                valPaths_img.append(image)
            
            for label in sorted(os.listdir(os.path.join(self.validation_path, "labels"))):
                valPaths_lbl.append(label)

        else: 
            trainPaths_img = imagePaths[self.val_size:]
            trainPaths_lbl = labelPaths[self.val_size:]

            valPaths_img = imagePaths[:self.val_size]
            valPaths_lbl = labelPaths[:self.val_size]
        
        return (trainPaths_img, trainPaths_lbl, valPaths_img, valPaths_lbl)

    def build(self):
        """This method builds the custom lists of paths.
        Call this by default, output can be used for either 
        TFRecord or HDF5 data preparation."""
        val_exist = self.val_isdir()

        (img_path, lbl_path) = self.get_path()

        (img_path, lbl_path) = self.shuffle(img_path, lbl_path)

        (trainPaths_img, trainPaths_lbl,
            valPaths_img, valPaths_lbl) = self.split_data(val_exist,
                                                          img_path, lbl_path)

        return (trainPaths_img, trainPaths_lbl, valPaths_img, valPaths_lbl)