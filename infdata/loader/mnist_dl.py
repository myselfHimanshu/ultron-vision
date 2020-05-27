"""
Data Loader Class
"""

import torch
from infdata.dataset.mnist_data import DownloadData

class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.num_workers = self.config["num_workers"]
        self.pin_memory = self.config["pin_memory"]
        self.batch_size = self.config["batch_size"]
        self.kwargs = {'num_workers':self.num_workers, "pin_memory":self.pin_memory} if self.config['use_cuda'] else {}

        ddata = DownloadData()
        self.train_loader = torch.utils.data.DataLoader(ddata.mnist_traindata, batch_size=self.batch_size,
                                                        shuffle=True, **self.kwargs)
        
        self.test_loader = torch.utils.data.DataLoader(ddata.mnist_testdata, batch_size=self.batch_size,
                                                        shuffle=True, **self.kwargs)
        

    