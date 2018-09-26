import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, n_inp, n_encoded):
        super(Autoencoder, self).__init__()
        self.encoder_fc1 = nn.Linear(n_inp, 1024)
        self.encoder_fc2 = nn.Linear(1024, n_encoded)

        self.decoder_fc1 = nn.Linear(n_encoded, 1024)
        self.decoder_fc2 = nn.Linear(1024, n_inp)

    def forward(self, x):
        e = F.sigmoid(self.encoder_fc1(x))
        e = F.sigmoid(self.encoder_fc2(e))

        d = self.decode(e)
        return e, d

    def decode(self, x):
        d = F.sigmoid(self.decoder_fc1(x))
        d = F.sigmoid(self.decoder_fc2(d))
        return d
