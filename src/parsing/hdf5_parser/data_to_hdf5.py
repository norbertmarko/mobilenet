import h5py
import os

class HDF5writer:
    def __init__(self, img_dims, label_dims, outputPath, buffSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already "
            "exists and cannot be overwritten. Manually delete "
            "the file before continuing.", outputPath)

        # Database
        self.db = h5py.File(outputPath, "w")
        # Datasets
        self.images = self.db.create_dataset("images", img_dims, dtype="float")
        self.labels = self.db.create_dataset("labels", label_dims, dtype="float")
        # Buffer
        self.buffSize = buffSize
        self.buffer = {"images": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        self.buffer["images"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["images"]) >= self.buffSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["images"]) 
        self.images[self.idx:i] = self.buffer["images"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"images": [], "labels": []}

    def close(self):
        if len(self.buffer["images"]) > 0:
            self.flush()
        self.db.close()
