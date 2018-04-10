import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.serialization import load_lua
from torchvision import datasets, transforms
import time, datetime
import numpy as np
import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace
import scipy.io as sio

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(21, 200)
        self.fc2 = nn.Linear(200, 500)
        self.drop1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(500, 7)
        # self.fc5 = nn.Linear(200, 100)
        # self.fc6 = nn.Linear(100, 7)
    def forward(self, x):
        x = x.view(-1, 21)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        return self.fc4(x)  


def train(epoch, xts, yts,xtr,ytr,hist):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.l1_loss(output, target)
        loss.backward()
        optimizer.step()
        #Total train Loss
        # datatr, targettr = Variable(xtr, volatile=True), Variable(ytr)
        # output = model(datatr)
        # train_loss = F.mse_loss(output, targettr).data[0] # sum up batch loss
        # hist['Ltr'].append(train_loss)
        hist['batch'].append(batch_idx)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    return hist

def test():
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target).float()
        output = model(data)
        test_loss += F.l1_loss(output, target).data[0] 
        
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

    

#Import Dataset
#N=5000
#data = np.random.uniform([-10,-10],[10,10],[N,2])
#labels = data[:,0]**2 + data[:,1]**2 + data[:,0]*data[:,1]
#data = torch.from_numpy(data).float()
#labels = torch.from_numpy(labels).float()

#t=int(0.1*N)
#xts = data[:t,:]
#yts = labels[:t]
#xtr = data[t:,:]
#ytr = labels[t:]

##
tr_data = sio.loadmat('/home/konstantinos/Documents/Intro_ML_Project/DNN/sarcos_inv.mat')
Ntr = tr_data['sarcos_inv'].shape[0]
xtr = tr_data['sarcos_inv'][:,:21]
ytr = tr_data['sarcos_inv'][:,21:]

xtr = torch.from_numpy(xtr).float()
ytr = torch.from_numpy(ytr).float()

ts_data = sio.loadmat('/home/konstantinos/Documents/Intro_ML_Project/DNN/sarcos_inv_test.mat')
Nts = ts_data['sarcos_inv_test'].shape[0]
xts = ts_data['sarcos_inv_test'][:,:21]
yts = ts_data['sarcos_inv_test'][:,21:]

xts = torch.from_numpy(xts).float()
yts = torch.from_numpy(yts).float()

##
trainset = torch.utils.data.TensorDataset(xtr,ytr)
testset = torch.utils.data.TensorDataset(xts,yts)

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4449, **kwargs)

#Define Model
model = None
model=LinearNet()
model.cuda()

#Configure optimizer
optimizer = optim.Adadelta(model.parameters())

#Train
start_time = datetime.datetime.now().replace(microsecond=0)
print('\n---Started training at---', (start_time))

epochs = 30
# hist = {'Ltr': [], 'batch': []}
hist = { 'batch': []}
    
for epoch in range(1, epochs + 1):
    hist = train(epoch,xts,yts,xtr,ytr,hist)
    test()
    current_time = datetime.datetime.now().replace(microsecond=0)
    print('Time Interval:', current_time - start_time, '\n')

# # for elem in ('Ltr', 'batch'):
# for elem in ( 'batch'):
#     hist[elem] = np.array(hist[elem])
 

#plt.figure(1)
#plt.plot(hist['Ltr'])
#plt.xlabel('Batches')
#plt.ylabel('Train Loss')


#plt.figure(2)
#plt.plot(hist['Lts'])
#plt.xlabel('Batches')
#plt.ylabel('Test Loss')

# plt.figure(3)
# plt.plot(hist['Ltr'])
# plt.plot(hist['Lts'])
# plt.xlabel('Batches')
# plt.ylabel('Loss')
# plt.legend(['Train','Test'])

# print('Final Training Loss: {}'.format(hist['Ltr'][-1]))
# print('Final Testing Loss: {}'.format(hist['Lts'][-1]))


