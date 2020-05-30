"""
Data Loader Class
"""

import torch
from infdata.dataset.cifar10_data import DownloadData

class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.num_workers = self.config["num_workers"]
        self.pin_memory = self.config["pin_memory"]
        self.batch_size = self.config["batch_size"]
        self.kwargs = {'num_workers':self.num_workers, "pin_memory":self.pin_memory} if self.config['use_cuda'] else {}

        ddata = DownloadData()
        self.train_loader = torch.utils.data.DataLoader(ddata.cifar10_traindata, batch_size=self.batch_size,
                                                        shuffle=True, **self.kwargs)
        
        self.valid_loader = torch.utils.data.DataLoader(ddata.cifar10_validdata, batch_size=self.batch_size,
                                                        shuffle=True, **self.kwargs)

        self.test_loader = torch.utils.data.DataLoader(ddata.cifar10_testdata, batch_size=1, shuffle=True, **self.kwargs)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.testclasses = tuple(ddata.testdata_classes.keys())

    