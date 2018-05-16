import time, datetime
import numpy as np

from Pipeline.option import args
from Pipeline.run import train, test, train_sw

start_time = datetime.datetime.now().replace(microsecond=0)
print('\n---Started training at---', (start_time))

mse = np.zeros([args.epochs,2])

train_fun = train
switch = 1

for epoch in range(1, args.epochs + 1):
    train_loss = train_fun(epoch)
    test_loss = test(epoch)

    current_time = datetime.datetime.now().replace(microsecond=0)
    print('Time Interval:', current_time - start_time, '\n')

    mse[epoch-1, 0] = train_loss
    mse[epoch-1, 1] = test_loss
    np.save('results/'+'mse_'+args.activation+'_l'+str(args.layers)+'_u'+str(args.units)+'.npy',mse)

    if (train_loss < args.threshold)&(switch == 1):
        train_fun = train_sw
        switch = 0
