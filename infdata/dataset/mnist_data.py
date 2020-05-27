"""
Define custom dataset function or download pre-defined data from pytorch
"""

from torchvision import datasets
from infdata.transformation.mnist_tf import Transforms
import os

curr_dir = os.path.dirname(__file__)

class DownloadData(object):
    def __init__(self):
        self.data_path = os.path.join(curr_dir,"../../","data")

        transf = Transforms()
        self.mnist_traindata = datasets.MNIST(self.data_path, train=True, download=True,
                                                transform=transf.train_transforms)
        
        self.mnist_validdata = datasets.MNIST(self.data_path, train=False, download=True,
                                                transform=transf.valid_transforms)

        self.mnist_testdata = None