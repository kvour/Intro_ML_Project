from torch.autograd import Variable
import torch.nn.functional as F

from Pipeline.option import args
from Data.data import train_loader, test_loader
from Architecture.model import model
from Architecture.optim import optimizer 

def train(epoch):
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
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

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
    return test_loss
