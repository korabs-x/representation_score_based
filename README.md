# Diffusion-Based Representation Learning

This repository contains the official implementation for the paper [Diffusion-Based Representation Learning](https://arxiv.org/abs/2105.14257).

Our code is built on top of the codebase of [Score-Based Generative Modeling through Stochastic Differential Equations](https://github.com/yang-song/score_sde_pytorch) and [LaplaceNet](https://github.com/psellcam/LaplaceNet).

## Using this repository
This repository contains the information to reproduce the results for representation learning and its application to semi-supervised image classification.

For data preparation and environment setup we refer to the descriptions in the above listed codebases, which can also be found in the readme files in the respective directories of this repository.

### Representation Learning
The following command learns a time-dependent encoding using a probabilistic encoder on CIFAR-10:<br>
`python3 main.py 
--config=configs/ve/cifar10_ncsnpp_small_continuous.py 
--workdir=/path/to/score_sde_pytorch 
--mode=train 
--config.data.dataset=cifar10 
--config.training.experiment_name=repr_cifar10
--config.training.include_encoder=True 
--config.training.probabilistic_encoder=True 
--config.training.lambda_z=1e-5 
--config.training.apply_mixup=False 
--config.training.lambda_reconstr=0.0
--config.training.n_iters=70000
--config.training.snapshot_freq=70000`

### Application to semi-supervised image classification (LaplaceNet)
To train the semi-supervised classification model starting from the pretrained encoder from above, execute the following command:<br>
`python3 main.py 
--num-labeled 100 
--ckpt ../score_sde_pytorch/checkpointsenc_ repr_cifar10/encoder_state_1.pth 
--mixup False 
--check_t False 
--dataset cifar10 
--load_ckpt 1 
--max_epochs 1000 
--model wrn-28-2 
--alpha 1.0 --lr 0.03 --labeled-batch-size 50 --batch-size 100 --aug-num 3 --label-split 12 --progress False`<br>
A checkpoint is updated after every epoch and training is resumed automatically when executing the same command again.
















