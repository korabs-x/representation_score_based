module load cuda/11.0
echo $LD_LIBRARY_PATH


# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=10000 --config.training.n_iters=140000 --config.eval.begin_ckpt=14 --config.eval.end_ckpt=14 --config.training.experiment_name=ve_c3_t60_kl1e-4_1 --config.model.frozen_encoder=True --config.model.lambda_z_sh=1e-4 --config.eval.enable_sampling=True
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar_large_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=10000 --config.training.n_iters=170000 --config.eval.begin_ckpt=17 --config.eval.end_ckpt=17 --config.training.experiment_name=ve_clarge_t60_kl1e-4_frozenstart3 --config.model.lambda_z_sh=1e-4 --config.model.frozen_encoder=True --config.eval.enable_sampling=True
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar_large_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=10000 --config.training.n_iters=50000 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=5 --config.training.experiment_name=ve_clarge_t$2_$1 --config.model.lambda_z_sh=1e-4 --config.model.single_t100=$2
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/cifar10_deep_continuous_vp.py --workdir=/home/kabstreiter/score_sde --mode=both --config.eval.num_samples=49999 --config.eval.begin_ckpt=20 --config.eval.end_ckpt=20 --config.training.n_iters=1000000 --config.training.experiment_name=pretrained_vp_c10_lambda_$2_$1 --config.training.lambda_method=$2

#XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar_large_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=10000 --config.training.n_iters=160001 --config.eval.begin_ckpt=16 --config.eval.end_ckpt=16 --config.training.experiment_name=c10_tenc_x0_lat$1 --config.model.time_dependent_encoder=True --config.model.predictx0=True --config.model.latent_input_dim=$1
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.eval.eval_dir_postfix=_intermed_n$2_$1 --config.eval.eval_intermediate_sampling_steps=True --config.eval.n_sampling_steps=$2 --config.eval.num_samples=49999 --config.eval.begin_ckpt=12 --config.eval.end_ckpt=12

# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.start_checkpoint_dir=checkpoints_pretrained_ve_c10 --config.training.experiment_name=pre_ve_sigmin$2_$1 --config.model.sigma_min=$2 --config.eval.num_samples=19999

#XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.start_checkpoint_dir=checkpoints_pretrained_ve_c10 --config.training.experiment_name=pre_ve_lm_rm$3_rs$4_sam-$2_$1 --config.training.lm_mean_reg=$3 --config.training.lm_std_reg=$4 --config.model.include_lambda_model=True --config.training.lambda_method_sampling=$2 --config.eval.num_samples=49999
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.start_checkpoint_dir=checkpoints_pretrained_ve_c10 --config.training.experiment_name=pre_ve_it1k_lmtype-$2_puni$3_$1 --config.training.lm_type=$2 --config.model.include_lambda_model=True --config.training.lambda_method_sampling=True --config.training.lm_basep=$3 --config.training.n_iters=800001 --config.training.snapshot_freq=50000 --config.eval.begin_ckpt=13 --config.eval.end_ckpt=15 --config.eval.enable_sampling=False
XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.experiment_name=pre_ve_it1k_lmtype-$2_puni$3_$1 --config.training.lm_type=$2 --config.model.include_lambda_model=True --config.training.lambda_method_sampling=True --config.training.lm_basep=$3 --config.training.n_iters=800001 --config.training.snapshot_freq=50000 --config.eval.begin_ckpt=13 --config.eval.end_ckpt=15 --config.eval.enable_sampling=False





