import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from Pipeline.option import args

t = args.timesteps
u = args.units

if args.activation == 'relu':
    act=F.relu
else:
    act=F.sigmoid

# batchnorm - relu -dropout

class LinearNet1(nn.Module):
    def __init__(self):
        super(LinearNet1, self).__init__()
        self.fc1 = nn.Linear(t*21, t*u)
        self.fc2 = nn.Linear(t*u, 7)
    def forward(self, x):
        x = x.view(-1, t*21)
        x = act(self.fc1(x))
        return self.fc2(x)

class LinearNet2(nn.Module):
    def __init__(self):
        super(LinearNet2, self).__init__()
        self.fc1 = nn.Linear(t*21, t*u)
        self.fc2 = nn.Linear(t*u, t*u)
        self.fc3 = nn.Linear(t*u, 7)
    def forward(self, x):
        x = x.view(-1, t*21)
        x = act(self.fc1(x))
        x = act(self.fc2(x))
        return self.fc3(x)

class LinearNet3(nn.Module):
    def __init__(self):
        super(LinearNet3, self).__init__()
        self.fc1 = nn.Linear(t*21, t*u)
        self.fc2 = nn.Linear(t*u, t*u)
        self.fc3 = nn.Linear(t*u, t*u)
        self.fc4 = nn.Linear(t*u, 7)
    def forward(self, x):
        x = x.view(-1, t*21)
        x = act(self.fc1(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        return self.fc4(x)

class LinearNet1b(nn.Module):
    def __init__(self):
        super(LinearNet1b, self).__init__()
        self.fc1 = nn.Linear(t*21, t*u)
        self.bn1 = nn.BatchNorm1d(t*u)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(t*u, 7)
    def forward(self, x):
        x = x.view(-1, t*21)
        x = self.fc1(x)
        x = self.bn1(x)
        x = act(x)
        x = self.drop1(x)
        return self.fc2(x)

class LinearNet2b(nn.Module):
    def __init__(self):
        super(LinearNet2b, self).__init__()
        self.fc1 = nn.Linear(t*21, t*u)
        self.bn1 = nn.BatchNorm1d(t*u)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(t*u, t*u)
        self.bn2 = nn.BatchNorm1d(t*u)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(t*u, 7)
    def forward(self, x):
        x = x.view(-1, t*21)
        x = self.fc1(x)
        x = self.bn1(x)
        x = act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = act(x)
        x = self.drop2(x)
        return self.fc3(x)

class LinearNet3b(nn.Module):
    def __init__(self):
        super(LinearNet3b, self).__init__()
        self.fc1 = nn.Linear(t*21, t*u)
        self.bn1 = nn.BatchNorm1d(t*u)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(t*u, t*u)
        self.bn2 = nn.BatchNorm1d(t*u)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(t*u, t*u)
        self.bn3 = nn.BatchNorm1d(t*u)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(t*u, 7)
    def forward(self, x):
        x = x.view(-1, t*21)
        x = self.fc1(x)
        x = self.bn1(x)
        x = act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = act(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = act(x)
        x = self.drop3(x)
        return self.fc4(x)


if args.layers == 1:
    if args.bndp == 1:
        model = LinearNet1b()
    else:
        model = LinearNet1()
elif args.layers == 2:
    if args.bndp == 1:
        model = LinearNet2b()
    else:
        model = LinearNet2()
elif args.layers == 3:
    if args.bndp == 1:
        model = LinearNet3b()
    else:
        model = LinearNet3()

if args.cuda:
    model.cuda()



print('\n---Model Information---')
print('Net:',model)
print('Use GPU:', args.cuda)
print('Activation: ',args.activation)