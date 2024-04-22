#!/bin/bash

# Define the lists of compression values and levels
compression_list=(0.5)
level_list=(2 3 4)

# Initial expid value
expid=30

# Loop over each compression value
for compression in "${compression_list[@]}"
do
    # Loop over each level
    for level in "${level_list[@]}"
    do
        # Execute the Python command with the current compression, level, and expid
        CUDA_VISIBLE_DEVICES=8 python main.py --model-class lottery \
            --model vgg16 --dataset cifar10 --experiment multishot --pruner mag \
            --compression-list $compression --level-list $level \
            --pre-epochs 50 --post-epochs 100 --expid $expid

        # Increment the expid for the next run
        ((expid++))
    done
done
