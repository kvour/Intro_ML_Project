# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Architecture/model.py: define model
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from Pipeline.option import args


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(21, 200)
        self.fc2 = nn.Linear(200, 500)
        self.drop1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 500)
        #self.drop2 = nn.Dropout(0.5)
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
        #x = self.drop2(x)
        return self.fc4(x)  


model = LinearNet()



if args.cuda:
    model.cuda()
    
print('\n---Model Information---')
print('Net:',model)
print('Use GPU:', args.cuda)
