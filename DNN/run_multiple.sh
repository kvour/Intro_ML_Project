#!/bin/bash

act='sigmoid relu'
lrate='1e-5 1e-4 1e-3 1e-2'
batchsize='64 128 256 512'
hiddenunits='200 300'
hiddenlayers='2 3'


for activation in $act
do
    for lr in $lrate
    do
        for bs in $batchsize
        do
            for hu in $hiddenunits
            do
                for hl in $hiddenlayers
                do
                    python3 train.py --epochs 500 --activation $activation --lr $lr --batch-size $bs --test-batch-size $bs --units $hu --layers $hl
                done
            done
        doneS
    done
done

python3 train.py --epochs 1000 --activation sigmoid --lr 0.0005 --batch-size 64 --test-batch-size 64 --units 1500 --layers 4
python3 train.py --epochs 1000 --activation sigmoid --lr 0.0005 --batch-size 64 --test-batch-size 64 --units 1500 --layers 5
python3 train.py --epochs 1500 --activation sigmoid --lr 0.0005 --batch-size 64 --test-batch-size 64 --units 1500 --layers 6
python3 train.py --epochs 1500 --activation sigmoid --lr 0.0001 --batch-size 64 --test-batch-size 64 --units 1500 --layers 6
