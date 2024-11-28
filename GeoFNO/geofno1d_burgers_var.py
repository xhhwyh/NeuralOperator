import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
sys.path.append("../")
from geofno_variant_mask import  GeoFNO, GeoFNO_train, compute_Fourier_modes

torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(10)

downsample_ratio = 8
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###################################
# load data
###################################
dataloader = loadmat("datasets\\burgers\\burgers_data_R10.mat")
data_in = np.array(dataloader.get('a'))
data_out = np.array(dataloader.get('u'))

print("data_in.shape " , data_in.shape)
print("data_out.shape", data_out.shape)

Np_ref = data_in.shape[1]
Np = 1 + (Np_ref -  1)//downsample_ratio
L = 1.0
grid_1d = np.linspace(0, L, Np)

data_in_ds = data_in[:, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio]


def get_weights(mask):
    Np = mask.shape[1]
    weights = np.ones(mask.shape) / Np
    target = []
    count = 0
    for n in range(mask.shape[0]):
        for i in range(Np):
            if mask[n, i] != 0:
                target.append(i)
                if len(target) == 2:
                    weights[n, target[0]] += count / 2
                    weights[n, target[1]] += count / 2
                    target = []
                    count = 0
                    target.append(i)
            else:
                count += 1 / Np
                weights[n, i] = 0
    return weights

# generate a random mask for each training data
mask_size = Np//2 + Np//4
mask = np.ones((n_train, Np))
for n in range(n_train):
    sequence = [1]*(mask_size-2) + [0]*(Np-mask_size)
    random.shuffle(sequence)
    mask[n][1:Np-1] = np.array(sequence)

# get the weights for each training data
weights = get_weights(mask)

# set the value on the dropped points to 0
data_in_ds[0:n_train,:] = data_in_ds[0:n_train,:] * mask
data_out_ds[0:n_train,:] = data_out_ds[0:n_train,:] * mask

# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.stack(
        (
            data_in_ds[0:n_train,:],
            np.tile(grid_1d, (n_train, 1)),
            weights,
            mask,
        ),
        axis=-1,
    ).astype(np.float32)
)
y_train = torch.from_numpy(data_out_ds[0:n_train, :, np.newaxis].astype(np.float32))

weights_uni = np.ones(Np) / Np
mask_uni = np.ones(Np)
x_test = torch.from_numpy(
    np.stack(
        (
            data_in_ds[-n_test:, :],
            np.tile(grid_1d, (n_test, 1)),
            np.tile(weights_uni, (n_test, 1)),
            np.tile(mask_uni, (n_test, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_test = torch.from_numpy(data_out_ds[-n_test:, :, np.newaxis].astype(np.float32))
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)

k_max = 32
ndim = 1
modes = compute_Fourier_modes(ndim, k_max, 1.0)
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoFNO(ndim, modes,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=2, out_dim=1,
               act='gelu').to(device)


epochs       = 1000
base_lr      = 0.001
scheduler    = "OneCycleLR"
weight_decay = 1.0e-4
batch_size   = 20

normalization_x = True
normalization_y = True
normalization_dim = []

config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim}}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoFNO_train(
    x_train, y_train, x_test, y_test, mask_size, config, model, save_model_name="GeoFNO_burgers.pth"
)