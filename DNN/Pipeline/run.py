from torch.autograd import Variable
import torch.nn.functional as F
import torch

from Pipeline.option import args
from Data.data import train_loader, test_loader, train_loader_sw
from Architecture.model import model
from Architecture.optim import optimizer, scheduler, scheduler_sw

def train(epoch):
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        scheduler.batch_step()
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, size_average=False)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
    if epoch % args.save_model_epoch==0:
        torch.save(model.state_dict(), 'results/'+'weights_'+args.activation+'_l'+str(args.layers)+'_u'+str(args.units)+'.pth')
    if epoch % 1 == 0:
        print('Epoch {} \n \nTrain set: Average Loss: {:.4f}'.format(epoch, train_loss/len(train_loader.dataset)))
    return train_loss/len(train_loader.dataset)

def train_sw(epoch):
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader_sw):
        scheduler_sw.batch_step()
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, size_average=False)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
    if epoch % args.save_model_epoch==0:
        torch.save(model.state_dict(), 'results/'+'weights_'+args.activation+'_l'+str(args.layers)+'_u'+str(args.units)+'.pth')
    if epoch % 1 == 0:
        print('Epoch {} \n \nTrain set: Average Loss: {:.4f}'.format(epoch, train_loss/len(train_loader_sw.dataset)))
    return train_loss/len(train_loader_sw.dataset)

def test(epoch):
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).float()
        output = model(data)
        test_loss += F.mse_loss(output, target, size_average=False).data[0]

    test_loss /= len(test_loader.dataset)
    if epoch % 10 == 0:
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))
        print('Test len = {}'.format(len(test_loader.dataset)))
    return test_loss
