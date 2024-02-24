# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 0.05 --expid 14 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 0.1 --expid 14 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 0.2 --expid 14 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 0.5 --expid 14 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 1 --expid 14 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 2 --expid 14 --post-epochs 100


# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 0.05 --expid 15 --post-epochs 100 --pre-epochs 200
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 0.1 --expid 15 --post-epochs 100 --pre-epochs 200
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 0.2 --expid 15 --post-epochs 100 --pre-epochs 200
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 0.5 --expid 15 --post-epochs 100 --pre-epochs 200
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 1 --expid 15 --post-epochs 100 --pre-epochs 200
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 2 --expid 15 --post-epochs 100 --pre-epochs 200

# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 0.05 --expid 16 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 0.1 --expid 16 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 0.2 --expid 16 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 0.5 --expid 16 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 1 --expid 16 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 2 --expid 16 --post-epochs 100

# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp --compression 0.05 --expid 17 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp --compression 0.1 --expid 17 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp --compression 0.2 --expid 17 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp --compression 0.5 --expid 17 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp --compression 1 --expid 17 --post-epochs 100
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp --compression 2 --expid 17 --post-epochs 100

# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 0.05 --expid 17 --post-epochs 100 > output_synflow_005.txt
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 0.1 --expid 17 --post-epochs 100 > output_synflow_01.txt
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 0.2 --expid 17 --post-epochs 100 > output_synflow_02.txt
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 0.5 --expid 17 --post-epochs 100 > output_synflow_05.txt
# python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 1 --expid 17 --post-epochs 100 > output_synflow_1.txt
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 2 --expid 17 --post-epochs 400 --verbose > output_synflow_2.txt
  