"""
MNIST MODEL ARCHITECTURE
date : 24th MAY 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1block = nn.Sequential(
            nn.Conv2d(1, 8, 3),                            #(-1,28,28,1)>(-1,3,3,1,8)>(-1,26,26,8)>3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3),                            #(-1,26,26,8)>(-1,3,3,8,8)>(-1,24,24,8)>5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 10, 3),                           #(-1,24,24,8)>(-1,3,3,8,10)>(-1,22,22,10)>7
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        self.pool1block = nn.Sequential(
            nn.MaxPool2d(2,2),                              #(-1,22,22,10)>(-1,11,11,10)>8
        )

        self.conv2block = nn.Sequential(
            nn.Conv2d(10, 16, 3),                           #(-1,11,11,10)>(-1,3,3,10,16)>(-1,9,9,16)>12
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv2d(16, 16, 3),                           #(-1,9,9,16)>(-1,3,3,16,16)>(-1,7,7,16)>16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv2d(16, 16, 3),                           #(-1,7,7,16)>(-1,3,3,16,16)>(-1,5,5,16)>20
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.01),
        )

        self.avgpool = nn.AvgPool2d(5)                      #(-1,5,5,16)>(-1,1,1,16)>28
        self.conv3 = nn.Conv2d(16, 10, 1)                   #(-1,1,1,16)>(-1,1,1,16,10)>(-1,1,1,10)>28  

        
    def forward(self, x):
        x = self.conv1block(x)
        x = self.pool1block(x)
        x = self.conv2block(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)