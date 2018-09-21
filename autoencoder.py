import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, n_inp, n_encoded):
        super(Autoencoder, self).__init__()
        self.encoder_fc1 = nn.Linear(n_inp, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        self.encoder_fc3 = nn.Linear(256, n_encoded)

        self.decoder_fc1 = nn.Linear(n_encoded, 256)
        self.decoder_fc2 = nn.Linear(256, 512)
        self.decoder_fc3 = nn.Linear(512, n_inp)

    def forward(self, x):
        e = F.relu(self.encoder_fc1(x))
        e = F.relu(self.encoder_fc2(e))
        e = F.relu(self.encoder_fc3(e))

        d = self.decode(e)
        return e, d

    def decode(self, x):
        d = F.relu(self.decoder_fc1(x))
        d = F.relu(self.decoder_fc2(d))
        d = F.sigmoid(self.decoder_fc3(d))
        return d
