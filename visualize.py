import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from autoencoder import Autoencoder
from gmmn import *
from constants import *

trans = transforms.Compose([transforms.ToTensor()])
test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)
view_data = [test_set[i][0] for i in range(N_ROWS * N_COLS)]

encoder_net = Autoencoder(N_INP, ENCODED_SIZE)
encoder_net.load_state_dict(torch.load(ENCODER_SAVE_PATH))

# for interactive mode
plt.ion()
plt.gray()

visualize_autoencoder = False
visualize_gmmn = True

if visualize_autoencoder:
    for i in range(N_ROWS * N_COLS):
        # original image
        r = i // N_COLS
        c = i % N_COLS + 1
        ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c)
        plt.imshow(view_data[i].squeeze())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # reconstructed image
        ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c + N_COLS)
        x = Variable(view_data[i])
        _, y = encoder_net(x.view(1, -1))
        plt.imshow(y.detach().squeeze().numpy().reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

elif visualize_gmmn:
    gmm_net = GMMN(NOISE_SIZE, ENCODED_SIZE)
    gmm_net.load_state_dict(torch.load(GMMN_SAVE_PATH))

    for r in range(N_ROWS):
        for c in range(N_COLS):
            ax = plt.subplot(N_ROWS, N_COLS, r * N_COLS + c + 1)

            noise = torch.rand((1, NOISE_SIZE)) * 2 - 1
            encoded_x = gmm_net(Variable(noise))
            y = encoder_net.decode(encoded_x)

            plt.imshow(y.detach().squeeze().numpy().reshape(28, 28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
plt.show()
plt.pause(20)
