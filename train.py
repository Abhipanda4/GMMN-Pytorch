import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from gmmn import *

is_visualize = False

root = "./data"
if not os.path.exists(root):
    os.mkdir(root)

model = "./model"
if not os.path.exists(model):
    os.mkdir(model)

ENCODER_SAVE_PATH = model + "/autoencoder.pth"
GMMN_SAVE_PATH = model + "/gmmn.pth"

BATCH_SIZE = 64
N_INP = 784
NOISE_SIZE = 64
ENCODED_SIZE = 128
N_ENCODER_EPOCHS = 20
N_GEN_EPOCHS = 50

# dataloader
trans = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)

test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=True)


# parameters for visualization
N_COLS = 8
N_ROWS = 4
view_data = [test_set[i][0] for i in range(N_ROWS * N_COLS)]

# train the autoencoder first
encoder_net = Autoencoder(N_INP, ENCODED_SIZE)
encoder_optim = optim.Adam(encoder_net.parameters())

if os.path.exists(ENCODER_SAVE_PATH):
    encoder_net.load_state_dict(torch.load(ENCODER_SAVE_PATH))
    print("Loaded saved autoencoder model")
else:
    for _ in range(N_ENCODER_EPOCHS):
        for idx, (x, _) in enumerate(train_loader):
            x = x.view(x.size()[0], -1)
            _, decoded = encoder_net(Variable(x))
            loss = torch.sum((x - decoded) ** 2)
            encoder_optim.zero_grad()
            loss.backward()
            encoder_optim.step()

        if not is_visualize:
            continue

        # for interactive mode
        plt.ion()
        plt.gray()

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
        plt.show()
        plt.pause(0.5)

    torch.save(encoder_net.state_dict(), ENCODER_SAVE_PATH)


print("Autoencoder has been successfully trained")
plt.ioff()

# define the GMMN
gmm_net = GMMN(NOISE_SIZE, ENCODED_SIZE)
if os.path.exists(GMMN_SAVE_PATH):
    gmm_net.load_state_dict(torch.load(GMMN_SAVE_PATH))
    print("Loaded previously saved GMM Network")

gmmn_optimizer = optim.Adam(gmm_net.parameters())

def get_scale_matrix(M, N):
    s1 = torch.ones((N, 1)) * 1.0 / N
    s2 = torch.ones((M, 1)) * -1.0 / M
    return torch.cat((s1, s2), 0)

def train_one_step(x, samples, sigma=[2, 5, 10, 20]):
    x = Variable(x)
    gen_samples = gmm_net(Variable(samples))
    X = torch.cat((gen_samples, x), 0)
    XX = torch.matmul(X, X.t())
    X2 = torch.sum(X * X, 1, keepdim=True)
    exp = XX - 0.5 * X2 - 0.5 * X2.t()

    M = gen_samples.size()[0]
    N = x.size()[0]
    s = get_scale_matrix(M, N)
    S = torch.matmul(s, s.t())

    loss = 0
    for v in sigma:
        kernel_val = torch.exp(1.0 / v * exp)
        loss += torch.sum(S * kernel_val)

    loss = torch.sqrt(loss)

    gmmn_optimizer.zero_grad()
    loss.backward()
    gmmn_optimizer.step()

    return loss

# training loop
for ep in range(N_GEN_EPOCHS):
    avg_loss = 0
    for idx, (x, _) in enumerate(train_loader):
        x = x.view(x.size()[0], -1)
        with torch.no_grad():
            encoded_x, _ = encoder_net(Variable(x))

        # uniform random noise between [-1, 1]
        random_noise = torch.rand((BATCH_SIZE, NOISE_SIZE)) * 2 - 1
        loss = train_one_step(encoded_x, random_noise)
        avg_loss += loss.item()

    avg_loss /= (idx + 1)
    print("Epoch [%3d] complete - avg loss was [%.4f]" %(ep, avg_loss))

    plt.ion()
    plt.gray()

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
    plt.pause(0.5)

torch.save(gmm_net.state_dict(), GMMN_SAVE_PATH)
