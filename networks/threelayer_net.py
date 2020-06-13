"""
3 layer Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

#define ghost batch Normalization
class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits=1, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

class BasicBlock(nn.Module):
    def __init__(self, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            GhostBatchNorm(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            GhostBatchNorm(planes),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.block(x)
        return out

class ThreeLayerNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(3LayerNet(), self).__init__()

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), # (32x32x3)(3x3x32x64)(32x32x64)
            GhostBatchNorm(64),
            nn.ReLU(),
        )

        self.x1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), # (32x32x64)(3x3x64x128)(32x32x128)
            nn.MaxPool2d(2,2),
            GhostBatchNorm(128),
            nn.ReLU(),
        ) 

        self.block1 = self._make_layer(block, 128, num_blocks[0], 1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), # (32x32x64)(3x3x64x128)(32x32x128)
            nn.MaxPool2d(2,2),
            GhostBatchNorm(256),
            nn.ReLU(),
        )

        self.x2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), # (32x32x64)(3x3x64x128)(32x32x128)
            nn.MaxPool2d(2,2),
            GhostBatchNorm(512),
            nn.ReLU(),
        )

        self.block2 = self._make_layer(block, 512, num_blocks[0], 1)

        self.pool_final = nn.MaxPool2d(4, 1)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] * num_blocks
        layers = []
        for stride in strides:
            layers.append(block(planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        prep_x = self.prep_layer(x)
        
        x1 = self.x1(x)
        block1 = self.block1(x1)
        x = x1 + block1

        x = self.layer2(x)
        
        x2 = self.x2(x)
        block2 = self.block2(x2)
        x = x2 + block2

        x = self.pool_final(x)
        x = x.view(-1, 512)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=-1)


def main():
    return ThreeLayerNet(BasicBlock, [1,1])




        
