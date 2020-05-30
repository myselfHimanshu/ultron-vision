import torch
import torch.nn as nn
import torch.nn.functional as F


class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()

        self.x0_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
        )
        self.x1_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
        )
        self.x2_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        
        self.p1_pool = nn.MaxPool2d(2, 2) 
        
        self.x4_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2), 
        )
        self.x5_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),  
        )
        self.x6_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        
        self.p2_pool = nn.MaxPool2d(2, 2) 
        
        self.x8_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),  
        )
        self.x9_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
        )
        self.x10_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.p3_gap = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.x12_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x1 = self.x0_blocl(x)
        x2 = self.x1_block(x1)
        x3 = self.x2_block(x1+x2)

        x4 = self.p1_pool(x1+x2+x3)

        x5 = self.x4_block(x4)
        x6 = self.x5_block(x4+x5)
        x7 = self.x6_block(x4+x5+x6)

        x8 = self.p2_pool(x5+x6+x7)

        x9 = self.x8_block(x8)
        x10 = self.x9_block(x8+x9)
        x11 = self.x10_block(x8+x9+x10)

        x12 = self.p3_gap(x11)
        x12 = x12.view(-1, 64)
        x13 = self.x12_fc(x12)

        return F.log_softmax(x13, dim=-1)