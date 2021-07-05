module load cuda/11.0
echo $LD_LIBRARY_PATH


# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --check_t 0 --mixup 0 --ckpt $1 --num-labeled $2 --dataset cifar10 --model wrn-28-2 --alpha 1.0 --lr 0.03 --labeled-batch-size 48 --batch-size 300 --aug-num 3 --label-split 12 --progress False
XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --num-labeled $2 --ckpt $3 --mixup $4 --check_t $5 --dataset $1 --model wrn-28-2 --alpha 1.0 --lr 0.03 --labeled-batch-size 50 --batch-size 100 --aug-num 3 --label-split 12 --progress False



