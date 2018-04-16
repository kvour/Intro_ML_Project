import time, datetime
from Pipeline.option import args
from Pipeline.run import train, test
from livelossplot import PlotLosses

start_time = datetime.datetime.now().replace(microsecond=0)
print('\n---Started training at---', (start_time))

liveloss = PlotLosses()

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    current_time = datetime.datetime.now().replace(microsecond=0)
    print('Time Interval:', current_time - start_time, '\n')
    liveloss.update({
        'test loss': test_loss,
    })

liveloss.draw()