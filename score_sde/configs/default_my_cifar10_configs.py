import ml_collections


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 32
    training.n_iters = 1300001
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
    training.train_clf = False

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
    evaluate.batch_size = 100
    evaluate.enable_sampling = False
    evaluate.conditional_sampling = False
    evaluate.guidance_factor = 1.
    evaluate.num_samples = 200
    evaluate.enable_loss = False
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'
    evaluate.eval_samples = False
    evaluate.enable_latent = True
    evaluate.enable_classifier = True
    evaluate.eval_intermediate_sampling_steps = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'CIFAR10'  # 'MNIST'  # 'CIFAR10'
    data.image_size = 28 if data.dataset == 'MNIST' else 32
    data.random_flip = True
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 1 if data.dataset == 'MNIST' else 3
    data.allowed_labels = None  # [0., 1., 2.]  # None  # [0, 1]
    data.allow_all_labels = False
    data.n_labelled_samples = -1  # 4000
    data.only_include_labeled_samples = False
    data.debug_data = False

    training.experiment_name = 've_cifarltd_sm_t09'  # ve_cifar_t08_L1e-5_samext ve_cifar_t08_L1e-5

    # model
    config.model = model = ml_collections.ConfigDict()
    model.single_t100 = "notset"
    model.lambda_z_sh = "notset"
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
    model.include_latent = model.include_latent_input
    model.lambda_z = None if not model.include_latent else (1e-5 if model.deterministic_latent_input else 1e-6)
    model.single_t = None  # .9  # .8
    model.same_xt_in_batch = False
    model.include_lambda_model = False
    model.uniform_std_sampling = False
    model.frozen_encoder = False
    model.time_dependent_encoder = False
    model.predict_x0 = False
    model.predictx0 = False
    model.lambda_reconstr = 0.
    model.lambda_reconstr_rate = 0.

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
