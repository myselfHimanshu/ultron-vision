"""
Data Loader Class
"""

import torch
from infdata.dataset.mnist_data import DownloadData

class DataLoader(object):
    def __init__(self, config, use_cuda):
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]
        self.batch_size = config["batch_size"]
        self.kwargs = {'num_workers':self.num_workers, "pin_memory":self.pin_memory} if use_cuda else {}

        self.train_loader = torch.utils.data.DataLoader(DownloadData.mnist_traindata, batch_size=self.batch_size,
                                                        shuffle=True, **self.kwargs)
        
        self.test_loader = torch.utils.data.DataLoader(DownloadData.mnist_testdata, batch_size=self.batch_size,
                                                        shuffle=True, **self.kwargs)
        

    