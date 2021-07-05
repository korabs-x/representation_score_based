module load cuda/11.0
echo $LD_LIBRARY_PATH


XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --num-labeled $1 --ckpt $2 --mixup $3 --check_t $4 --dataset cifar10 --model wrn-28-2 --alpha 1.0 --lr 0.03 --labeled-batch-size 50 --batch-size 100 --aug-num 3 --label-split 12 --progress False


