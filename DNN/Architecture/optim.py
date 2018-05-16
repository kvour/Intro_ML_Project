import torch.optim as optim

from Pipeline.option import args
from Architecture.model import model
from Architecture.cls import CyclicLR

optimizer = None
if args.optimizer =='SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr2min, momentum=0.5)
elif args.optimizer =='Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr2min)
elif args.optimizer =='Adadelta':
    optimizer = optim.Adadelta(model.parameters())
else:
    raise ValueError('Wrong name of optimizer')

scheduler = CyclicLR(optimizer, base_lr = args.lr1min, max_lr = args.lr1max, step_size =args.cli1)
scheduler_sw = CyclicLR(optimizer, base_lr = args.lr2min, max_lr = args.lr2max, step_size =args.cli2)
