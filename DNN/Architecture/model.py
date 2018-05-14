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

#Batchnorm before activation

class LinearNet1(nn.Module):
    def __init__(self):
        super(LinearNet1, self).__init__()
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

class LinearNet2(nn.Module):
    def __init__(self):
        super(LinearNet2, self).__init__()
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



class LinearNet3(nn.Module):
    def __init__(self):
        super(LinearNet3, self).__init__()
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

class LinearNet4(nn.Module):
    def __init__(self):
        super(LinearNet4, self).__init__()
        self.fc1 = nn.Linear(t*21, t*500)
        self.bn1 = nn.BatchNorm1d(t*500)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(t*500, t*1000)
        self.bn2 = nn.BatchNorm1d(t*1000)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(t*1000, t*1000)
        self.bn3 = nn.BatchNorm1d(t*1000)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(t*1000, t*500)
        self.bn4 = nn.BatchNorm1d(t*500)
        self.drop4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(t*500, 7)
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
        x = self.fc4(x)
        x = self.bn4(x)
        x = act(x)
        x = self.drop4(x)
        return self.fc5(x)


class LinearNet5(nn.Module):
    def __init__(self):
        super(LinearNet5, self).__init__()
        self.fc1 = nn.Linear(t*21, t*600)
        self.bn1 = nn.BatchNorm1d(600*u)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(t*600, t*1500)
        self.bn2 = nn.BatchNorm1d(1500*u)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(t*1500, t*1500)
        self.bn3 = nn.BatchNorm1d(1500*u)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(t*1500, t*600)
        self.bn4 = nn.BatchNorm1d(600*u)
        self.drop4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(t*600, 7)
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
        x = self.fc4(x)
        x = self.bn4(x)
        x = act(x)
        x = self.drop4(x)
        return self.fc5(x)

class LinearNet6(nn.Module):
    def __init__(self):
        super(LinearNet6, self).__init__()
        self.fc1 = nn.Linear(t*21, t*600)
        self.fc2 = nn.Linear(t*600, t*1500)
        self.fc3 = nn.Linear(t*1500, t*1500)
        self.fc4 = nn.Linear(t*1500, t*600)
        self.fc5 = nn.Linear(t*600, 7)
    def forward(self, x):
        x = x.view(-1, t*21)
        x = act(self.fc1(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        x = act(self.fc4(x))
        return self.fc5(x)

if args.layers == 1:
    model = LinearNet1()
elif args.layers == 2:
    model = LinearNet2()
elif args.layers == 3:
    model = LinearNet3()
elif args.layers == 4:
    model = LinearNet4()
elif args.layers == 5:
    model = LinearNet5()
else:
    model = LinearNet6()



if args.cuda:
    model.cuda()



print('\n---Model Information---')
print('Net:',model)
print('Use GPU:', args.cuda)
print('Activation: ',args.activation)