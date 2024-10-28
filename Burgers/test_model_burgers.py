import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
sys.path.append("../")
from geofno import  GeoFNO, GeoFNO_train, compute_Fourier_modes, UnitGaussianNormalizer
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.losses import LpLoss

torch.set_printoptions(precision=16)

torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_mask(grid: int, rate: float):
    # generate a random mask by dropping points on a given rate
    mask = np.ones(grid)
    for k in range(1, grid-1):
        rand = np.random.random(1)
        if rand < rate:
            mask[k] = 0

    # compute weights based on the mask
    weights = np.ones(grid) / grid
    target = []
    count = 0
    for i in range(grid):
        if mask[i] != 0:
            target.append(i)
            if len(target) == 2:
                weights[target[0]] += count / 2
                weights[target[1]] += count / 2
                target = []
                count = 0
                target.append(i)
        else:
            count += 1 / grid
            weights[i] = 0
    return (weights, mask)

def train(weights, mask):
    loss_1024 = 0
    loss_794  = 0
    loss_487  = 0
    loss_217  = 0
    for n in range(1900, 1901):
        n_test = 1

        # get the nth data from the dataset
        x_test = torch.from_numpy(
            np.stack(
                (
                    data_in_ds[n, :].reshape(1, data_in_ds.shape[1]),
                    np.tile(grid_1d, (n_test, 1)),
                    np.tile(weights, (n_test, 1)),
                    np.tile(mask, (n_test, 1)),
                ),
                axis=-1,
            ).astype(np.float32)
        )
        y_test = torch.from_numpy(data_out_ds[n, :, np.newaxis].reshape(1, data_out_ds.shape[1], 1).astype(np.float32))

        # take the mask
        xf = x_test
        ind = list(np.nonzero(mask)[0])
        x_test = x_test[:, ind, :]
        yf = y_test
        y_test = y_test[:, ind, :]

        x_normalizer = UnitGaussianNormalizer(x_test, aux_dim=3)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = UnitGaussianNormalizer(y_test, aux_dim=0)
        y_test = y_normalizer.encode(y_test)

        # load models which is trianed on size of 1024, 794, 487, 217 points respectively
        model_1024 = torch.load("Burgers\models\GeoFNO_size1024.pth", map_location = device, weights_only=False)
        model_794  = torch.load("Burgers\models\GeoFNO_size794.pth",  map_location = device, weights_only=False)
        model_487  = torch.load("Burgers\models\GeoFNO_size487.pth",  map_location = device, weights_only=False)
        model_217  = torch.load("Burgers\models\GeoFNO_size217.pth",  map_location = device, weights_only=False)

        # make predictions by the four models
        y_1024 = model_1024(x_test, xf)
        y_794  = model_794(x_test, xf)
        y_487  = model_487(x_test, xf)
        y_217  = model_217(x_test, xf)

        y_1024 = y_normalizer.decode(y_1024)
        y_794  = y_normalizer.decode(y_794)
        y_487  = y_normalizer.decode(y_487)
        y_217  = y_normalizer.decode(y_217)

        # get the true result
        y_test = y_normalizer.decode(y_test)

        # compute loss
        myloss = LpLoss(d=1, p=2, size_average=False)
        loss_1024 += myloss(y_1024.view(1, -1), y_test.view(1, -1))
        loss_794  += myloss(y_794.view(1, -1), y_test.view(1, -1))
        loss_487  += myloss(y_487.view(1, -1), y_test.view(1, -1))
        loss_217  += myloss(y_217.view(1, -1), y_test.view(1, -1))

    print("loss_1024: ", loss_1024.item())
    print("loss_794:  ", loss_794.item())
    print("loss_487:  ", loss_487.item())
    print("loss_217:  ", loss_217.item())

    x_grid = np.linspace(0, L, Np)
    x_grid = x_grid[ind]
    
    trueyf = yf[0, :, 0].detach().numpy()
    pred_1024 = y_1024[0, :, 0].detach().numpy()
    pred_794  = y_794[0, :, 0].detach().numpy()
    pred_487  = y_487[0, :, 0].detach().numpy()
    pred_217  = y_217[0, :, 0].detach().numpy()
    
    return x_grid, trueyf, pred_1024, pred_794, pred_487, pred_217

###################################
# load data
###################################
dataloader = loadmat("datasets\\burgers\\burgers_data_R10.mat")
data_in = np.array(dataloader.get('a'))
data_out = np.array(dataloader.get('u'))

print("data_in.shape " , data_in.shape)
print("data_out.shape", data_out.shape)

downsample_ratio = 8

data_in_ds = data_in[:, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio]

rates = [0, 0.2, 0.5, 0.8]
sizes = []
x_grids = []
trueyfs = []
pred1024s = []
pred794s  = []
pred487s  = []
pred217s  = []
for rate in rates:
    print("----------------------------------------------------------------------------------")
    Np_ref = data_in.shape[1]
    Np = 1 + (Np_ref -  1)//downsample_ratio
    L = 1.0
    grid_1d = np.linspace(0, L, Np)
    weights, mask = get_mask(grid_1d.shape[0], rate)
    size = np.count_nonzero(mask)
    sizes.append(size)
    print(size)
    print(mask)
    x_grid, true, init, test2, test3, test4 = train(weights, mask)
    x_grids.append(x_grid)
    trueyfs.append(true)
    pred1024s.append(init)
    pred794s.append(test2)
    pred487s.append(test3)
    pred217s.append(test4)

# plot
x_fullgrid = np.linspace(0, L, Np)

fig, axs = plt.subplots(1, 4, figsize = (12, 3))
for i in range(4):
    axs[i].plot(x_fullgrid, trueyfs[i], label='true')
    axs[i].plot(x_grids[i], pred1024s[i], label='size1024')
    axs[i].plot(x_grids[i], pred794s[i], label='size794')
    axs[i].plot(x_grids[i], pred487s[i], label='size487')
    axs[i].plot(x_grids[i], pred217s[i], label='size217')
    axs[i].set_title(f'rate = {rates[i]}, size = {sizes[i]}', y = -0.25, fontsize = 12)
    axs[i].tick_params(axis='x', labelsize=8)
    axs[i].tick_params(axis='y', labelsize=8)
    axs[i].legend(fontsize = 8)
    axs[i].set_ylim([-1.5, 2])

plt.tight_layout(pad=1, w_pad=1)
plt.show()
