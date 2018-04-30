import torch
import argparse

parser = argparse.ArgumentParser(description='AML Project')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training')

parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing')

parser.add_argument('--optimizer',default='Adam', metavar='OPTM',
                    help='SGD, Adam or Adadelta')

parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train')

parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')

parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='batches to wait before logging training status')

parser.add_argument('--save_model_epoch', type=int, default=250, metavar='S',
                    help='epochs to wait before saving')

parser.add_argument('--timesteps', type=int, default=1, metavar='T',
                    help='number of input timesteps')

parser.add_argument('--units', type=int, default=300, metavar='U',
                    help='number of hidden units')

parser.add_argument('--layers', type=int, default=6, metavar='L',
                    help='number of hidden layers (different network for 5/6)')

parser.add_argument('--activation', default='relu', metavar='A',
                    help='activation function: relu/sigmoid')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()