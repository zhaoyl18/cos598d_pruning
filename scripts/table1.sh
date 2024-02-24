python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 1 --expid 4 --post-epochs 100

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 1 --expid 5 --post-epochs 100 --pre-epochs 200

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 1 --expid 6 --post-epochs 100

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp  --compression 1 --expid 7 --post-epochs 400 --verbose

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 1 --expid 8 --post-epochs 100