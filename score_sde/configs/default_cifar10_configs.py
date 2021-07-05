import ml_collections


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 32
    training.n_iters = 650001  # 600000
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = False
    training.likelihood_weighting = False
    training.continuous = True
    training.n_jitted_steps = 5
    training.reduce_mean = False
    training.lambda_method = ''
    training.lambda_method_sampling = False  # whether to sample or weight
    training.start_checkpoint_dir = ''
    training.p_eps = 0.0
    # lambda model params
    # training.include_lambda_model = False # WARNING: SET in model!!!
    training.lm_type = 'mog'
    training.lm_input = 't'
    training.lm_n_mog = 10
    training.lm_mean_reg = 1e-2
    training.lm_std_reg = 1e-3
    training.lm_basep = 0.01

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 13
    evaluate.end_ckpt = 13
    evaluate.batch_size = 50  # 1024
    evaluate.enable_sampling = True
    evaluate.num_samples = 20000  # 50000
    evaluate.enable_loss = False  # True
    evaluate.enable_bpd = False  # True
    evaluate.bpd_dataset = 'test'

    evaluate.enable_latent = True
    evaluate.n_sampling_steps = 1000
    evaluate.eval_dir_postfix = ''
    evaluate.conditional_sampling = False
    evaluate.guidance_factor = 1.
    evaluate.eval_intermediate_sampling_steps = False
    evaluate.use_vae_samples = False
    evaluate.vae_sample_t = 0.0
    evaluate.vae_sample_is_p0 = True
    config.training.experiment_name = 'pretrained_ve_c10'

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'CIFAR10'  # 'MNIST'  # 'CIFAR10'
    data.image_size = 28 if data.dataset == 'MNIST' else 32
    data.random_flip = True
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 1 if data.dataset == 'MNIST' else 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    model.include_lambda_model = False  # training.include_lambda_model

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
