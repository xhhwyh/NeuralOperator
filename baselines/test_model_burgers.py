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

downsample_ratio = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_weights(grid: int, rate):
    weights = np.ones(grid) / grid
    mask = np.ones(grid)
    for k in range(1, grid-1):
        rand = np.random.random(1)
        if rand < rate:
            mask[k] = 0
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
    loss_init = 0
    loss_test2 = 0
    loss_test3 = 0
    loss_test4 = 0
    for n in range(1900, 1901):
        n_test = 1

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

        xf = x_test
        ind = list(np.nonzero(mask)[0])
        x_test = x_test[:, ind, :]
        yf = y_test
        y_test = y_test[:, ind, :]

        x_normalizer = UnitGaussianNormalizer(x_test, aux_dim=3)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = UnitGaussianNormalizer(y_test, aux_dim=0)
        y_test = y_normalizer.encode(y_test)

        model_init = torch.load("baselines\model\geofno1d_burgers\GeoFNO_burgers_init.pth", map_location = device, weights_only=False)
        model_test2 = torch.load("baselines\model\geofno1d_burgers\GeoFNO_burgers_test2.pth", map_location = device, weights_only=False)
        model_test3 = torch.load("baselines\model\geofno1d_burgers\GeoFNO_burgers_test3.pth", map_location = device, weights_only=False)
        model_test4 = torch.load("baselines\model\geofno1d_burgers\GeoFNO_burgers_test4.pth", map_location = device, weights_only=False)

        y_init = model_init(x_test, xf)
        y_test2 = model_test2(x_test, xf)
        y_test3 = model_test3(x_test, xf)
        y_test4 = model_test4(x_test, xf)

        y_init = y_normalizer.decode(y_init)
        y_test2 = y_normalizer.decode(y_test2)
        y_test3 = y_normalizer.decode(y_test3)
        y_test4 = y_normalizer.decode(y_test4)

        y_test = y_normalizer.decode(y_test)

        myloss = LpLoss(d=1, p=2, size_average=False)
        loss_init += myloss(y_init.view(1, -1), y_test.view(1, -1))
        loss_test2 += myloss(y_test2.view(1, -1), y_test.view(1, -1))
        loss_test3 += myloss(y_test3.view(1, -1), y_test.view(1, -1))
        loss_test4 += myloss(y_test4.view(1, -1), y_test.view(1, -1))

    print("loss_init:  ", loss_init.item())
    print("loss_test2: ", loss_test2.item())
    print("loss_test3: ", loss_test3.item())
    print("loss_test4: ", loss_test4.item())

    trueyf = yf[0, :, 0].detach().numpy()
    init = y_init[0, :, 0].detach().numpy()
    test2 = y_test2[0, :, 0].detach().numpy()
    test3 = y_test3[0, :, 0].detach().numpy()
    test4 = y_test4[0, :, 0].detach().numpy()

    x_grid = np.linspace(0, L, Np)
    x_grid = x_grid[ind]
    
    return x_grid, trueyf, init, test2, test3, test4

###################################
# load data
###################################
dataloader = loadmat("datasets\\burgers\\burgers_data_R10.mat")
data_in = np.array(dataloader.get('a'))
data_out = np.array(dataloader.get('u'))

print("data_in.shape " , data_in.shape)
print("data_out.shape", data_out.shape)

data_in_ds = data_in[:, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio]

rates = [0, 0.2, 0.5, 0.8]
sizes = []
x_grids = []
trueyfs = []
inits = []
test2s = []
test3s = []
test4s = []
for rate in rates:
    print("----------------------------------------------------------------------------------")
    Np_ref = data_in.shape[1]
    Np = 1 + (Np_ref -  1)//downsample_ratio
    L = 1.0
    grid_1d = np.linspace(0, L, Np)
    weights, mask = get_weights(grid_1d.shape[0], rate)
    size = np.count_nonzero(mask)
    sizes.append(size)
    print(size)
    print(mask)
    x_grid, true, init, test2, test3, test4 = train(weights, mask)
    x_grids.append(x_grid)
    trueyfs.append(true)
    inits.append(init)
    test2s.append(test2)
    test3s.append(test3)
    test4s.append(test4)

# plot
x_fullgrid = np.linspace(0, L, Np)

fig, axs = plt.subplots(1, 4, figsize = (12, 3))
for i in range(4):
    axs[i].plot(x_fullgrid, trueyfs[i], label='true')
    axs[i].plot(x_grids[i], inits[i], label='full_grid')
    axs[i].plot(x_grids[i], test2s[i], label='ds_grid0.2')
    axs[i].plot(x_grids[i], test3s[i], label='ds_grid0.5')
    axs[i].plot(x_grids[i], test4s[i], label='ds_grid0.8')
    axs[i].set_title(f'rate = {rates[i]}, size = {sizes[i]}', y = -0.25, fontsize = 12)
    axs[i].tick_params(axis='x', labelsize=8)
    axs[i].tick_params(axis='y', labelsize=8)
    axs[i].legend(fontsize = 8)
    axs[i].set_ylim([-1.5, 2])

plt.tight_layout(pad=1, w_pad=1)
plt.show()
