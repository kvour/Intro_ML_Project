import numpy as np
import torch
from torchvision import datasets, transforms
from Pipeline.option import args
import scipy.io as sio

def split_data(data, timesteps=1):
    X, Y = np.split(data, [21], axis=1)

    if timesteps == 1:
        return X, Y

    Xres = np.zeros((X.shape[0] - timesteps + 1, 21 * timesteps))
    Yres = Y[timesteps - 1:]

    for i in range(X.shape[0] - timesteps + 1):
        Xres[i] = np.reshape(X[i:i+timesteps], -1)

    return Xres, Yres


tr_data = sio.loadmat('../data/sarcos_inv.mat')['sarcos_inv']
xtr, ytr = split_data(tr_data, args.timesteps)

#Normalize
xtr = (xtr - np.mean(xtr,0)[None,:])/np.std(xtr,0)[None,:]
ytr = (ytr - np.mean(ytr,0)[None,:])/np.std(ytr,0)[None,:]

xtr = torch.from_numpy(xtr).float()
ytr = torch.from_numpy(ytr).float()


ts_data = sio.loadmat('../data/sarcos_inv_test.mat')['sarcos_inv_test']
xts, yts = split_data(ts_data, args.timesteps)

#Normalize
xts = (xts - np.mean(xts,0)[None,:])/np.std(xts,0)[None,:]
yts = (yts - np.mean(yts,0)[None,:])/np.std(yts,0)[None,:]

xts = torch.from_numpy(xts).float()
yts = torch.from_numpy(yts).float()


trainset = torch.utils.data.TensorDataset(xtr,ytr)
testset = torch.utils.data.TensorDataset(xts,yts)

kwargs = {'num_workers': 8, 'pin_memory': True, 'drop_last': False}
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.init_batch_size, shuffle = True, **kwargs)
train_loader_sw = torch.utils.data.DataLoader(trainset, batch_size=args.sw_batch_size, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle = True, **kwargs)
