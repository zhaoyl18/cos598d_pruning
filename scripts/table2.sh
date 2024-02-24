# python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner rand --compression 1 --expid 9 --post-epochs 20

# python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner mag --compression 1 --expid 10 --pre-epochs 20 --post-epochs 10

# python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner snip --compression 1 --expid 11 --post-epochs 30

# python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner grasp  --compression 1 --expid 12 --post-epochs 30

python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner synflow --compression 1 --expid 13 --post-epochs 150 --verbose > output_MNIST_synflow_2.txt