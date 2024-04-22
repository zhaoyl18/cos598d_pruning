#!/bin/bash

# Define the lists of compression values and levels
compression_list=(0.05 0.1 0.2 0.5 1.0 2.0)

# Initial expid value
expid=1

# Loop over each compression value
for compression in "${compression_list[@]}"
do
    pruner=rand
    # Execute the Python command with the current compression, level, and expid
    CUDA_VISIBLE_DEVICES=9 python main.py --model-class lottery \
        --model vgg16 --dataset cifar10 --experiment singleshot --pruner $pruner \
        --compression $compression --post-epochs 100 --expid $expid

    ((expid++))
done

for compression in "${compression_list[@]}"
do
    pruner=mag
    # Execute the Python command with the current compression, level, and expid
    CUDA_VISIBLE_DEVICES=9 python main.py --model-class lottery \
        --model vgg16 --dataset cifar10 --experiment singleshot --pruner $pruner \
        --compression $compression --post-epochs 100 --pre-epochs 200 --expid $expid

    ((expid++))
done

for compression in "${compression_list[@]}"
do
    pruner=snip
    # Execute the Python command with the current compression, level, and expid
    CUDA_VISIBLE_DEVICES=9 python main.py --model-class lottery \
        --model vgg16 --dataset cifar10 --experiment singleshot --pruner $pruner \
        --compression $compression --post-epochs 100 --expid $expid

    ((expid++))
done

for compression in "${compression_list[@]}"
do
    pruner=grasp
    # Execute the Python command with the current compression, level, and expid
    CUDA_VISIBLE_DEVICES=9 python main.py --model-class lottery \
        --model vgg16 --dataset cifar10 --experiment singleshot --pruner $pruner \
        --compression $compression --post-epochs 400 --expid $expid

    ((expid++))
done

for compression in "${compression_list[@]}"
do
    pruner=synflow
    # Execute the Python command with the current compression, level, and expid
    CUDA_VISIBLE_DEVICES=9 python main.py --model-class lottery \
        --model vgg16 --dataset cifar10 --experiment singleshot --pruner $pruner \
        --compression $compression --post-epochs 100 --expid $expid

    ((expid++))
done
