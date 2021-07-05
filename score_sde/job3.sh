module load cuda/11.0
echo $LD_LIBRARY_PATH


# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/toydata_deep_continuous_subvp.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=1024 --config.training.snapshot_freq=10000 --config.training.n_iters=200001 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=10 --config.eval.batch_size=1024 --config.eval.num_samples=1024
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/toydata_deep_continuous_subvp.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=1024 --config.training.snapshot_freq=25000 --config.training.n_iters=100001 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=4 --config.eval.batch_size=1024 --config.eval.num_samples=1024
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/aedata_deep_continuous_subvp.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.batch_size=1024 --config.training.snapshot_freq=100000 --config.training.n_iters=100001 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=1 --config.eval.batch_size=10000
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/mnist_deep_continuous_subvp.py --workdir=/home/kabstreiter/score_sde --mode=train --config.training.batch_size=128 --config.training.snapshot_freq=10000 --config.training.n_iters=100001 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=1 --config.eval.enable_sampling=True --config.eval.batch_size=128 --config.eval.num_samples=128
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/mnist_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=128 --config.training.snapshot_freq=10000 --config.training.n_iters=80001 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=8 --config.eval.batch_size=128 --config.eval.num_samples=128
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/mnist_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.batch_size=128 --config.training.snapshot_freq=10000 --config.training.n_iters=160001 --config.eval.begin_ckpt=8 --config.eval.end_ckpt=8 --config.eval.denoise_samples_wrong_z=False --config.eval.enable_sampling=True --config.model.frozen_encoder=False --config.model.latent_input_dim=$1 --config.training.experiment_name=ve_m10_lat$1d_tuni_kl1e-7
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.experiment_name=pretrained_ve_c10_lambda_$2_$1 --config.training.lambda_method=$2
XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar_small_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=80000 --config.training.n_iters=80001 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=1 --config.training.experiment_name=c10_tclfAEsq --config.model.time_dependent_encoder=True --config.model.latent_input_dim=20 --config.model.lambda_reconstr_rate=-1e0



# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 train_mnist_classifier.py
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 train_mnist_ae.py
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 train_mnist_vae.py
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 train_mnist_vae_prob.py
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 train_cifar_ae.py


