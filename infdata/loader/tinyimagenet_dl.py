"""
Data Loader Class
"""

import torch
from infdata.dataset.tinyimagenet_data import DownloadData

class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.num_workers = self.config["num_workers"]
        self.pin_memory = self.config["pin_memory"]
        self.batch_size = self.config["batch_size"]
        self.kwargs = {'num_workers':self.num_workers, "pin_memory":self.pin_memory} if self.config['use_cuda'] else {}

        ddata = DownloadData()

        self.train_loader = torch.utils.data.DataLoader(ddata.tinyimagenet_traindata, batch_size=self.batch_size,
                                                        shuffle=True, **self.kwargs)
        
        self.valid_loader = torch.utils.data.DataLoader(ddata.tinyimagenet_validdata, batch_size=self.batch_size,
                                                        shuffle=True, **self.kwargs)

        
        self.classes2idx = ddata.classes2idx
        

    