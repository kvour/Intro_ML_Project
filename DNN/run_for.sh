#!/bin/bash

act='relu sigmoid'
hiddenunits='100 200 300'
hiddenlayers='1 2 3'


for activation in $act
do
    for hu in $hiddenunits
    do
        for hl in $hiddenlayers
        do
            python3 train.py --epochs 700 --activation $activation --units $hu --layers $hl --bndp 1
        done
    done
done
