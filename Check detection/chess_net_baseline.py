import torch
from torch import nn
from torch.nn import functional as F

class chess_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,5,padding=2)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,128,5,padding=2)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.fc1 = nn.Linear(128*2*2,512)
        self.fc2 = nn.Linear(512,1)
        self.droupout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(x.shape[0],128*2*2)
        x = self.droupout(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x