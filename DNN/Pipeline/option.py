import torch
import argparse

parser = argparse.ArgumentParser(description='AML Project')

parser.add_argument('--init-batch-size', type=int, default=44484, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--sw-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--test-batch-size', type=int, default=4449, metavar='N',
                    help='input batch size for testing (default: 64)')

parser.add_argument('--threshold', type=int, default=2, metavar='T',
                    help='switching threshold')

parser.add_argument('--optimizer',default='Adam', metavar='OPTM',
                    help='define optimizer (default: Adam)')

parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr1min', type=float, default=0.005, metavar='LR',
                    help='min cycling learning rate before switch')

parser.add_argument('--lr1max', type=float, default=0.01, metavar='LR',
                    help='max cycling learning rate before switch')

parser.add_argument('--lr2min', type=float, default=0.00005, metavar='LR',
                    help='min cycling learning rate after switch')

parser.add_argument('--lr2max', type=float, default=0.0001, metavar='LR',
                    help='max cycling learning rate after switch')

parser.add_argument('--cli1', type=int, default=4, metavar='CL',
                    help='CLR iterations pes half cycle before switch')

parser.add_argument('--cli2', type=int, default=6000, metavar='CL',
                    help='CLR iterations pes half cycle after switch')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--bndp', type=int, default=1, metavar='DP',
                    help='Batchnorm and Dropout')

parser.add_argument('--save_model_epoch', type=int, default=1000, metavar='S',
                    help='epochs to wait before saving')

parser.add_argument('--timesteps', type=int, default=1, metavar='T',
                    help='number of input timesteps')

parser.add_argument('--units', type=int, default=300, metavar='U',
                    help='number of hidden units')

parser.add_argument('--layers', type=int, default=3, metavar='L',
                    help='number of hidden layers')

parser.add_argument('--activation', default='sigmoid', metavar='A',
                    help='activation function')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
