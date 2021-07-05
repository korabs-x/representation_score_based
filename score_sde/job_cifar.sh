module load cuda/11.0
echo $LD_LIBRARY_PATH


#XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=10000 --config.training.n_iters=70000 --config.eval.begin_ckpt=7 --config.eval.end_ckpt=7 --config.training.experiment_name=ve_c3_t$2_kl1e-4_$1 --config.model.single_t100=$2 --config.model.lambda_z_sh=1e-4
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar_large_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=10000 --config.training.n_iters=100000 --config.eval.begin_ckpt=6 --config.eval.end_ckpt=10 --config.training.experiment_name=ve_clarge_tstd_$1 --config.model.lambda_z_sh=1e-4
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar_large_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.batch_size=64 --config.training.snapshot_freq=10000 --config.training.n_iters=100000 --config.eval.begin_ckpt=10 --config.eval.end_ckpt=10 --config.training.experiment_name=ve_c10_t$2_det$3_lam$4_$1 --config.model.deterministic_latent_input=$3 --config.model.single_t100=$2 --config.model.lambda_z_sh=$4 --config.data.allow_all_labels=True --config.eval.enable_sampling=True
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar_large_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=10000 --config.training.n_iters=300000 --config.eval.begin_ckpt=30 --config.eval.end_ckpt=30 --config.training.experiment_name=ve_c10struc_tstd$2_det$3_lam$4_$1 --config.model.deterministic_latent_input=$3 --config.model.uniform_std_sampling=$2 --config.model.lambda_z_sh=$4 --config.eval.enable_sampling=True --config.data.allow_all_labels=True
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar_large_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=10000 --config.training.n_iters=1000001 --config.eval.begin_ckpt=100 --config.eval.end_ckpt=100 --config.training.experiment_name=c10_tenc_lat$1 --config.model.time_dependent_encoder=True --config.model.latent_input_dim=$1
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/mnist_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=128 --config.training.snapshot_freq=10000 --config.training.n_iters=160001 --config.eval.begin_ckpt=16 --config.eval.end_ckpt=16 --config.model.time_dependent_encoder=True
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/cifar10_deep_continuous_vp.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.eval.num_samples=49999 --config.eval.begin_ckpt=19 --config.eval.end_ckpt=19 --config.training.experiment_name=pretrained_vp_c10

# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ddpmpp/my_cifar_large_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.batch_size=64 --config.training.snapshot_freq=80000 --config.training.n_iters=80001 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=1 --config.training.experiment_name=c10_tencrec_lat$1_lam$2_3 --config.model.time_dependent_encoder=True --config.model.lambda_reconstr=$2 --config.model.latent_input_dim=$1
# Pure eval:
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.eval.eval_dir_postfix=_n$2_$1 --config.eval.n_sampling_steps=$2 --config.eval.num_samples=49999 --config.eval.begin_ckpt=12 --config.eval.end_ckpt=12
# One checkpoint of training, then eval:
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.start_checkpoint_dir=checkpoints_pretrained_ve_c10 --config.training.experiment_name=pretrained_ve_c10_lambda_$2_$1 --config.training.lambda_method=$2 --config.eval.num_samples=19999
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=both --config.training.start_checkpoint_dir=checkpoints_pretrained_ve_c10 --config.training.experiment_name=pre_ve_lm_rm$3_rs$4_puni$2_$1 --config.training.lm_mean_reg=$3 --config.training.lm_std_reg=$4 --config.model.include_lambda_model=True --config.training.lambda_method_sampling=True --config.training.lm_basep=$2 --config.eval.num_samples=20000
XLA_FLAGS=--xla_gpu_cuda_data_dir=/is/software/nvidia/cuda-11.0/ /usr/bin/python3 main.py --config=configs/ncsnpp/cifar10_deep_continuous_ve.py --workdir=/home/kabstreiter/score_sde --mode=eval --config.training.start_checkpoint_dir=checkpoints_pretrained_ve_c10 --config.training.experiment_name=pre_ve_lmtype-$2_puni$3_$1 --config.training.lm_type=$2 --config.model.include_lambda_model=True --config.training.lambda_method_sampling=True --config.training.lm_basep=$3






