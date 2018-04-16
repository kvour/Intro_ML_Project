# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Data/data.py: load data and process data
# read: http://pytorch.org/docs/master/data.html
# data loaders are *iterators*!
import torch
from torchvision import datasets, transforms
from Pipeline.option import args
import scipy.io as sio

tr_data = sio.loadmat('../data/sarcos_inv.mat')
Ntr = tr_data['sarcos_inv'].shape[0]
xtr = tr_data['sarcos_inv'][:,:21]
ytr = tr_data['sarcos_inv'][:,21:]

xtr = torch.from_numpy(xtr).float()
ytr = torch.from_numpy(ytr).float()

ts_data = sio.loadmat('../data/sarcos_inv_test.mat')
Nts = ts_data['sarcos_inv_test'].shape[0]
xts = ts_data['sarcos_inv_test'][:,:21]
yts = ts_data['sarcos_inv_test'][:,21:]

xts = torch.from_numpy(xts).float()
yts = torch.from_numpy(yts).float()

##
trainset = torch.utils.data.TensorDataset(xtr,ytr)
testset = torch.utils.data.TensorDataset(xts,yts)

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, **kwargs)
