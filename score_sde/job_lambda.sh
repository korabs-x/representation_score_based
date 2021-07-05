module load cuda/11.0
echo $LD_LIBRARY_PATH


XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/toydata_deep_continuous_subvp.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.batch_size=1024 --config.training.snapshot_freq=5000 --config.training.n_iters=200001 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.batch_size=10000 --config.eval.num_samples=10000
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/toydata_deep_continuous_subvp.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.batch_size=1024 --config.training.snapshot_freq=5000 --config.training.n_iters=200001 --config.eval.begin_ckpt=10 --config.eval.end_ckpt=10 --config.eval.batch_size=1000 --config.eval.num_samples=100


