import ml_collections


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128
    training.n_iters = 1300001
    training.snapshot_freq = 50000
    training.log_freq = 50  # 5000  # 50
    training.eval_freq = 100  # 10000
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True  # True
    training.likelihood_weighting = False
    training.continuous = True
    training.n_jitted_steps = 5
    training.reduce_mean = False
    training.experiment_name = 'binmnist'  # mnistsmall mnist_3d_reg0 mnist_2d_reg1e-5 mnist_128d_nondet_reg1e-5 mnist_10d_reg1e-5_norm05 mnist_latent_reg128_diffarc mnist_latent_noreg128_diffarc mnist_latent_noreg10_diffarc mnist_latent_noreg10 mnist mnist_latent

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 128
    evaluate.enable_sampling = True
    evaluate.num_samples = 50000
    evaluate.enable_loss = False
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'
    evaluate.eval_samples = False
    evaluate.enable_latent = True

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'binMNIST'  # 'CIFAR10'
    data.image_size = 28
    data.random_flip = False
    data.centered = True
    data.uniform_dequantization = False
    data.num_channels = 1

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    model.include_latent_input = True
    model.deterministic_latent_input = True  # True
    model.latent_input_dim = 2
    model.include_latent = True
    model.lambda_z = None if not model.include_latent else (0 if model.deterministic_latent_input else 0.)

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42

    return config
