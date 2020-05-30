"""
Define custom dataset function or download pre-defined data from pytorch
"""

from torchvision import datasets
from infdata.transformation.cifar10_tf import Transforms
import os

curr_dir = os.path.dirname(__file__)

class DownloadData(object):
    def __init__(self):
        self.data_path = os.path.join(curr_dir,"../../","data")

        transf = Transforms()
        self.cifar10_traindata = datasets.CIFAR10(self.data_path, train=True, download=True,
                                                transform=transf.train_transforms)
        
        self.cifar10_validdata = datasets.CIFAR10(self.data_path, train=False, download=True,
                                                transform=transf.valid_transforms)

        self.cifar10_testdata = datasets.ImageFolder(root=os.path.join(self.data_path, "test-images"), 
                                                transform=transf.test_transforms)

        self.testdata_classes = self.cifar10_testdata.class_to_idx