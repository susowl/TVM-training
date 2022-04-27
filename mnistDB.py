"""
    Simple MNIST dataloader.
    This code based on:
        https://github.com/hsjeong5/MNIST-for-Numpy

    Returns:
        _type_: _description_
"""


import numpy as np
from urllib import request
import gzip
import pickle
import os

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

class mnistDB:
    def __init__(self,path="./DB/") -> None:
        self.path = path
        if not os.path.exists(path):
            os.makedirs(self.path)
        path = os.path.join(path,"MNIST.pkl")
        if os.path.exists(path):
            self.mnistDB = self.load()
        else:
            self.download_mnist()
            self.save_mnist()
            self.mnistDB = self.load()
        

    def download_mnist(self):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in filename:
            print("Downloading "+name[1]+"...")
            request.urlretrieve(base_url+name[1], os.path.join(self.path,name[1]))
        print("Download complete.")

    def save_mnist(self):
        mnist = {}
        for name in filename[:2]:
            with gzip.open(os.path.join(self.path,name[1]), 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
        for name in filename[-2:]:
            with gzip.open(os.path.join(self.path,name[1]), 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(os.path.join(self.path,"MNIST.pkl"), 'wb') as f:
            pickle.dump(mnist,f)
        print("Save complete.")

    def load(self):
        with open(os.path.join(self.path,"MNIST.pkl"),'rb') as f:
            mnist = pickle.load(f)
        return mnist

    def trainDataloader(self,batchsize=32):
        for i in range(0,len(self.mnistDB['training_images']),batchsize):
            yield self.mnistDB['training_images'][i:i+batchsize],self.mnistDB['training_labels'][i:i+batchsize]

    def testDataloader(self,batchsize=32):
        for i in range(0,len(self.mnistDB[0]),batchsize):
            yield self.mnistDB['test_images'][i:i+batchsize],self.mnistDB['test_images'][i:i+batchsize]

