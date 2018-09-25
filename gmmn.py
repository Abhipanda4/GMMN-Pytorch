import torch
import torch.nn as nn
import torch.nn.functional as F

class GMMN(nn.Module):
    def __init__(self, n_start, n_inp):
        super(GMMN, self).__init__()
        self.fc1 = nn.Linear(n_start, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, n_inp)

    def forward(self, samples):
        x = F.sigmoid(self.fc1(samples))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x
