python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 0.5 --expid 24 --post-epochs 100 > output_rand.txt

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 0.5 --expid 25 --post-epochs 100 --pre-epochs 200 > output_mag.txt

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 0.5 --expid 26 --post-epochs 100 > output_snip.txt

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp  --compression 0.5 --expid 27 --post-epochs 300 --verbose > output_grasp.txt

python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 0.5 --expid 28 --post-epochs 100 > output_synflow.txt