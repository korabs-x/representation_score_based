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
    training.snapshot_freq_for_preemption = 100000000
    ## produce samples at each snapshot.
    training.snapshot_sampling = False
    training.likelihood_weighting = False
    training.continuous = True
    training.n_jitted_steps = 5
    training.reduce_mean = False
    training.experiment_name = 'latent_toy1_1e-4'  # lambda_toy3_lambdalatent_tsampling lambda_toy3_latentnolambda lambda_toy3_latentlambdax0 lambda_toy3_lambdalatent_highfreq # lambda_toy3_std1e-3 lambda_toy3_nolambda lambda_toy2_std1e-3_temp3 lambda_toy2_2 lambda_toy2_nolambda lambda_latent lambda_mogstdlong x0z0_2dim_reg2 lambdanolambda x0z0_2dim lambdaplus1 lambdaminus1e1 lambdaminus0 lambdaminus1e-3 lambdaminus1e-1

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16
    sampling.clip_to_255 = False

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 1024
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = False
    evaluate.enable_bpd = False  # True
    evaluate.bpd_dataset = 'test'
    evaluate.eval_samples = False
    evaluate.enable_latent = True
    evaluate.enable_latentgrid = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'toydatav1'  # 'CIFAR10'
    data.image_size = 1
    data.random_flip = False
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 2

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 20.  # 0.1  # 0.1
    model.beta_max = 0.1  # 20.  # 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    model.include_latent_input = True  # True
    model.deterministic_latent_input = False  # True
    model.include_latent = True  # True
    model.latent_input_dim = 1
    model.lambda_z = None if not model.include_latent else (0. if model.deterministic_latent_input else 1e-4)
    model.include_lambda_model = False

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42

    return config
