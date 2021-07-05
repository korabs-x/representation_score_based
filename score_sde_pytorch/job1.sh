module load cuda/11.0
#module load cuda/8.0
#echo $LD_LIBRARY_PATH
#echo $PATH
#echo $CUDA_HOME


# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/
/usr/bin/python3 main.py --config=configs/ve/cifar10_ncsnpp_small_continuous.py --workdir=/home/kabstreiter/score_sde_pytorch --mode=train --config.training.include_encoder=True --config.training.experiment_name=$1
# /usr/bin/python3 test.py --config=configs/ve/cifar10_ncsnpp_small_continuous.py --workdir=/home/kabstreiter/score_sde_pytorch --mode=train
# /usr/bin/python3 test.py
