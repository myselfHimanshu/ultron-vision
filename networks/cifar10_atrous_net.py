"""
CIFAR10 Atrous network MODEL ARCHITECTURE
date : 24th MAY 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Cifar10AtrousNet(nn.Module):
    def __init__(self):
        super(Cifar10AtrousNet, self).__init__()

        self.conv1block = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),     #(-1,32,32,3)>(-1,3,3,3,32)>(-1,32,32,32)>3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),    #(-1,32,32,32)>(-1,3,3,32,64)>(-1,32,32,64)>5
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dconv1block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=2, bias=False, dilation=2),   #(-1,32,32,3)>(-1,3,3,32,64)>(-1,32,32,64)>5
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv11block = nn.Sequential(
            depthwise_separable_conv(64, 32),               #(-1,32,32,64)>(-1,32,32,32)>7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.pool1block = nn.Sequential(
            nn.MaxPool2d(2,2),                              #(-1,32,32,32)>(-1,16,16,32)>8
        )

        self.conv2block = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),     #(-1,16,16,32)>(-1,3,3,32,64)>(-1,16,16,64)>12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),     #(-1,16,16,64)>(-1,3,3,64,64)>(-1,16,16,64)>16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dconv2block = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=2, bias=False, dilation=2),   #(-1,16,16,32)>(-1,3,3,32,64)>(-1,16,16,64)>16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv22block = nn.Sequential(
            depthwise_separable_conv(64, 64),                #(-1,16,16,64)>(-1,16,16,64)>20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.pool2block = nn.Sequential(
            nn.MaxPool2d(2,2),                              #(-1,16,16,64)>(-1,8,8,64)>22
        )


        self.conv3block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),     #(-1,8,8,64)>(-1,3,3,64,64)>(-1,8,8,64)>30
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),    #(-1,8,8,64)>(-1,3,3,64,128)>(-1,8,8,128)>38
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=0, bias=False),    #(-1,8,8,128)>(-1,3,3,128,128)>(-1,6,6,128)>46
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.avgpool = nn.AvgPool2d(6)                      #(-1,6,6,128)>(-1,1,1,128)>50
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),              #(-1,1,1,128)>(-1,1,1,128,64)>(-1,1,1,64)>66
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fcblock = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        
    def forward(self, x):
        x1 = self.conv1block(x)
        x2 = self.dconv1block(x)
        x = torch.add(x1, x2)
        x = self.conv11block(x)
        x = self.pool1block(x)
        
        x3 = self.conv2block(x)
        x4 = self.dconv2block(x)
        x = torch.add(x3, x4)
        x = self.conv22block(x)
        x = self.pool2block(x)

        x = self.conv3block(x)
        x = self.avgpool(x)
        x = self.conv4(x)
        x = x.view(-1, 64)
        x = self.fcblock(x)
        return F.log_softmax(x, dim=-1)