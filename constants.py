root = "./data"
model = "./models"

ENCODER_SAVE_PATH = model + "/autoencoder.pth"
GMMN_SAVE_PATH = model + "/gmmn.pth"

BATCH_SIZE = 100
N_INP = 784
NOISE_SIZE = 10
ENCODED_SIZE = 128
N_ENCODER_EPOCHS = 20
N_GEN_EPOCHS = 100

# visualization
N_COLS = 8
N_ROWS = 4
