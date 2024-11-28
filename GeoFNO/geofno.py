import math
import numpy as np
import torch
import sys
from timeit import default_timer
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.adam import Adam
from models.losses import LpLoss
## FNO 1D and 2D

class UnitGaussianNormalizer(object):
    def __init__(self, x, aux_dim = 0, eps=1.0e-5):
        super(UnitGaussianNormalizer, self).__init__()
        # x: ndata, nx, nchannels
        # when dim = [], mean and std are both scalars
        self.aux_dim = aux_dim
        self.mean = torch.mean(x[...,0:x.shape[-1]-aux_dim])
        self.std = torch.std(x[...,0:x.shape[-1]-aux_dim])
        self.eps = eps

    def encode(self, x):
        x[...,0:x.shape[-1]-self.aux_dim] = (x[...,0:x.shape[-1]-self.aux_dim] - self.mean) / (self.std + self.eps)
        return x
    

    def decode(self, x):
        std = self.std + self.eps # n
        mean = self.mean
        x[...,0:x.shape[-1]-self.aux_dim] = (x[...,0:x.shape[-1]-self.aux_dim] * std) + mean
        return x
    
    
    def to(self, device):
        if device == torch.device('cuda:0'):
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        else:
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()



def _get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    elif act == "none":
        func = None
    else:
        raise ValueError(f"{act} is not supported")
    return func



@torch.jit.script
def compl_mul1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    res = torch.einsum("bix,iox->box", a, b)
    return res


@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res = torch.einsum("bixy,ioxy->boxy", a, b)
    return res


def compute_Fourier_modes(ndim, nks, Ls):
    # 1d
    if ndim == 1:
        n = nks
        L = Ls
        nk = 2*n + 1
        k_pairs = np.zeros(nk)
        for k in range(-n, n+1):
            k_pairs[k] = 2*np.pi/L*k
        k_pairs = k_pairs[np.argsort(k_pairs), np.newaxis]
    # 2d
    elif ndim == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2*nx*ny + nx + ny
        k_pairs    = np.zeros((nk, ndim))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if (ky==0 and kx<=0): 
                    continue
                k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1
        k_pairs = k_pairs[np.argsort(k_pair_mag), :]
    return k_pairs

def compute_Fourier_bases(grid, modes, mask):
    # grid : batchsize, ndim, nx (8, 2, 14641)
    # modes: nk, ndim (544, 2)
    # mask : batchsize, 1, nx (8, 1, 14641)
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temp  = torch.einsum("bdx,kd->bkx", grid, modes)
    #temp: batchsize, nx, nk
    bases_c = torch.cos(temp)
    bases_s = torch.sin(temp)
    bases_0 = torch.ones(mask.shape).to(device)
    return bases_c, bases_s, bases_0

################################################################
# 2d fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        nmode, ndim = modes.shape
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, nmode, dtype=torch.float))
        self.weights_s = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, nmode, dtype=torch.float))
        self.weights_0 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 1, dtype=torch.float))


    def forward(self, x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0):
        size = x.shape[-1] # 14641

        x_c_hat =  torch.einsum("bix,bkx->bik", x, wbases_c) # (8, 128, 544)
        x_s_hat = -torch.einsum("bix,bkx->bik", x, wbases_s) # (8, 128, 544)
        x_0_hat =  torch.einsum("bix,bkx->bik", x, wbases_0) # (8, 128, 1)

        weights_c, weights_s, weights_0 = self.weights_c/(size), self.weights_s/(size), self.weights_0/(size)

        f_c_hat = torch.einsum("bik,iok->bok", x_c_hat, weights_c) - torch.einsum("bik,iok->bok", x_s_hat, weights_s)
        f_s_hat = torch.einsum("bik,iok->bok", x_s_hat, weights_c) + torch.einsum("bik,iok->bok", x_c_hat, weights_s)
        f_0_hat = torch.einsum("bik,iok->bok", x_0_hat, weights_0)

        x = torch.einsum("bok,bkx->box", f_0_hat, bases_0)  + 2*torch.einsum("bok,bkx->box", f_c_hat, bases_c) -  2*torch.einsum("bok,bkx->box", f_s_hat, bases_s) 

        return x
    
class GeoFNO(nn.Module):
    def __init__(
        self,
        ndim,
        modes,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
    ):
        super(GeoFNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0.
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2.
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes = modes
        
        self.layers = layers
        self.fc_dim = fc_dim

        self.ndim = ndim
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv2d(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)

    def forward(self, x, xf):
        """
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        """
        length = len(self.ws)

        batch_size = x.shape[0]

        aux = xf[...,-2-self.ndim:].permute(0, 2, 1)    # coord, weights, mask
        grid, weights, mask = aux[:, 0:self.ndim, :], aux[:, -2:-1, :], aux[:, -1:, :]
        # grid: (8, 2, 14641)
        # weights: (8, 1, 14641)
        # mask: (8, 1, 14641)

        bases_c, bases_s, bases_0 = compute_Fourier_bases(grid, self.modes, mask)
        # bases_c: (8, 544, 14641)
        # bases_s: (8, 544, 14641)
        # bases_0: (8,  1 , 14641)
        
        size = int(torch.count_nonzero(mask[0, 0, :]))
        wbases_c, wbases_s, wbases_0 = bases_c*(weights*size), bases_s*(weights*size), bases_0*(weights*size)

        x = self.fc0(x[...,0:self.in_dim])
        x = x.permute(0, 2, 1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ind = list(torch.nonzero(mask[0, 0, :]).squeeze().detach().cpu().numpy())
        xf = torch.zeros((batch_size, self.fc_dim, xf.shape[1])).to(device)
        xf[:, :, ind] = x

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(xf, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0)
            x1 = x1[:, :, ind]
            x2 = w(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x

# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def GeoFNO_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./GeoFNO_model"):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y, normalization_dim = config["train"]["normalization_x"], config["train"]["normalization_y"], config["train"]["normalization_dim"]
    ndim = model.ndim # n_train, size, n_channel
    print("In GeoFNO_train, ndim = ", ndim)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_train_fullgrid = x_train
    x_test_fullgrid = x_test
    mask = x_train[0, :, -1]
    ind = list(torch.nonzero(mask).squeeze().detach().cpu().numpy())
    x_train = x_train[:, ind, :]
    y_train = y_train[:, ind, :]
    x_test = x_test[:, ind, :]
    y_test = y_test[:, ind, :]

    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, aux_dim = ndim+2)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
    
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, aux_dim = 0)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, x_train_fullgrid, y_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test_fullgrid, y_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    
    # Load from checkpoint
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
    if config['train']['scheduler'] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    elif config['train']['scheduler'] == "CosineAnnealingLR":
        T_max = (config['train']['epochs']//10)*(n_train//config['train']['batch_size'])
        eta_min  = 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min = eta_min)
    elif config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['train']['base_lr'], 
            div_factor=2, final_div_factor=100,pct_start=0.2,
            steps_per_epoch=1, epochs=config['train']['epochs'])
    elif config["train"]["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    else:
        print("Scheduler ", config['train']['scheduler'], " has not implemented.")

    model.train()
    myloss = LpLoss(d=1, p=2, size_average=False)

    epochs = config['train']['epochs']


    for ep in range(epochs):
        t = 0
        t1 = default_timer()
        train_rel_l2 = 0

        model.train()
        for x, xf, y in train_loader:
            x, xf, y = x.to(device), xf.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, xf) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0
        with torch.no_grad():
            for x, xf, y in test_loader:
                x, xf, y = x.to(device), xf.to(device), y.to(device)
                batch_size_ = x.shape[0]
                out = model(x, xf) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()

        scheduler.step()

        train_rel_l2/= n_train
        test_l2 /= n_test
        test_rel_l2/= n_test
        
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)

        t2 = default_timer()
        t += t2 - t1
        if (ep % 10 == 0) or (ep == epochs -1):
            t_avg = t / 10
            t = 0
            print("Epoch : ", ep, "Train time : ", t_avg," Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2, flush=True)
            torch.save(model, save_model_name)
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses