# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any
from collections import defaultdict, Counter

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import functools
from flax.metrics import tensorboard
from flax.training import checkpoints
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, mnistmodel, mnistsmall, toymodel, lambdamodel, cifarmodel
import losses
import sampling
import controllable_generation
import utils
from models import utils as mutils
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import sys
import pickle
import train_mnist_classifier as mnist_clf
import cv2
import scipy.stats as sps
import matplotlib.pyplot as plt
from utils import batch_mul

FLAGS = flags.FLAGS


def shouldBeOptimized(path, x):
    # print(("path:", path), file=sys.stderr)
    # print('Encoder_0' not in path, file=sys.stderr)
    return 'Encoder' not in path


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    print("tf.config.list_physical_devices('GPU')")
    print(tf.config.list_physical_devices('GPU'))
    print("tf.config.list_logical_devices('GPU')")
    print(tf.config.list_logical_devices('GPU'))
    logging.info(('jax.local_device_count()', jax.local_device_count()))

    print("Create directories for experimental logs")
    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    rng = jax.random.PRNGKey(config.seed)
    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    if jax.host_id() == 0:
        writer = tensorboard.SummaryWriter(tb_dir)

    print("Initialize model")
    x0_input = hasattr(config.model, 'include_latent_input') and config.model.include_latent_input
    # Initialize model.
    rng, step_rng = jax.random.split(rng)
    print(("x0_input:", x0_input), file=sys.stderr)
    logging.info(("deterministic_latent_input:", getattr(config.model, 'deterministic_latent_input', None)))
    logging.info(("tstdsampling:", getattr(config.model, 'uniform_std_sampling', None)))
    logging.info(("lambda_z:", getattr(config.model, 'lambda_z', None)))
    if hasattr(config.data, "allowed_labels"):
        logging.info(("allowed_labels:", config.data.allowed_labels))
    score_model, init_model_state, initial_params = mutils.init_model(step_rng, config, x0_input=x0_input)

    frozen_encoder = hasattr(config.model, 'frozen_encoder') and config.model.frozen_encoder
    focus = None
    if frozen_encoder:
        focus = flax.optim.ModelParamTraversal(shouldBeOptimized)
    optimizer = losses.get_optimizer(config).create(initial_params, focus=focus)

    rng, step_rng = jax.random.split(rng)
    lambda_model, lambda_init_model_state, lambda_initial_params, lambda_optimizer = None, None, None, None
    use_lambda_model = hasattr(config.model, 'include_lambda_model') and config.model.include_lambda_model
    logging.info(f'use_lambda_model={use_lambda_model}')
    if use_lambda_model:
        lambda_model, lambda_init_model_state, lambda_initial_params = mutils.init_lambda_model(step_rng, config)
        lambda_optimizer = losses.get_optimizer(config).create(lambda_initial_params)

    lambda_reconstr_rate = getattr(config.model, 'lambda_reconstr_rate', 0.0)
    lambda_method = getattr(config.training, 'lambda_method', '')
    lambda_reconstr_balanced = None
    if lambda_reconstr_rate > 0 or lambda_method == 'lossprop':
        lambda_reconstr_balanced = jnp.asarray([1.0] * 20)

    start_checkpoint_dir = getattr(config.training, 'start_checkpoint_dir', '')
    state_new = mutils.State(step=0,
                             optimizer=optimizer,
                             lr=config.optim.lr,
                             model_state=init_model_state,
                             ema_rate=config.model.ema_rate,
                             params_ema=initial_params,
                             lambda_step=0,
                             lambda_optimizer=lambda_optimizer,
                             lambda_lr=1e-1,
                             lambda_model_state=lambda_init_model_state,
                             lambda_ema_rate=config.model.ema_rate,
                             lambda_params_ema=lambda_initial_params,
                             lambda_reconstr_balanced=lambda_reconstr_balanced,
                             rng=rng)  # pytype: disable=wrong-keyword-args

    old_state = 'pretrained' in config.training.experiment_name or 'pretrained' in start_checkpoint_dir  # config.data.image_size > 1
    if old_state:
        state = mutils.Stateold(step=0,
                                optimizer=optimizer,
                                lr=config.optim.lr,
                                model_state=init_model_state,
                                ema_rate=config.model.ema_rate,
                                params_ema=initial_params,
                                rng=rng)  # pytype: disable=wrong-keyword-args
    else:
        state = state_new

    print("Create checkpoints directory")

    eval_dir = os.path.join(workdir, 'eval')
    if hasattr(config.training, 'experiment_name'):
        eval_dir += f'_{config.training.experiment_name}'
    if hasattr(config.eval, 'eval_dir_postfix'):
        eval_dir += f'{config.eval.eval_dir_postfix}'
    tf.io.gfile.makedirs(eval_dir)
    if True or (lambda_reconstr_rate > 0. or lambda_method == 'lossprop'):
        lambda_reconstr_balanced_dir = os.path.join(eval_dir, 'lambda_reconstr_balanced')
        tf.io.gfile.makedirs(lambda_reconstr_balanced_dir)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
    if hasattr(config.training, 'experiment_name'):
        checkpoint_dir += f'_{config.training.experiment_name}'
        checkpoint_meta_dir += f'_{config.training.experiment_name}'
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(checkpoint_meta_dir)
    # if start_checkpoint_dir != '':
    #    tf.io.gfile.copy(os.path.join(workdir, start_checkpoint_dir), os.path.join(checkpoint_meta_dir, 'checkpoint_1'))
    # Resume training when intermediate checkpoints are detected
    state = checkpoints.restore_checkpoint(checkpoint_meta_dir if start_checkpoint_dir == '' else start_checkpoint_dir,
                                           state)
    state = state_new.replace(step=state.step,
                              optimizer=state.optimizer,
                              lr=state.lr,
                              model_state=state.model_state,
                              ema_rate=state.ema_rate,
                              params_ema=state.params_ema,
                              rng=state.rng)
    del state_new
    if frozen_encoder:
        state = state.replace(
            optimizer=losses.get_optimizer(config).create(state.params_ema, focus=focus)
        )

    # `state.step` is JAX integer on the GPU/TPU devices
    initial_step = int(state.step)
    rng = state.rng

    print("Build data iterators")
    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=config.training.n_jitted_steps,
                                                uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    print("Setup SDEs")
    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    if hasattr(config.model, 'single_t') and config.model.single_t is not None:
        std = sde.marginal_prob(jnp.asarray([0.]), jnp.asarray([config.model.single_t]))[1]
        print(f'SINGLE_T={config.model.single_t}, STD={round(std, 5)}', file=sys.stderr)
        for t in [1e-3, 1e-2, 1e-1, 0.2, 0.5, 0.8]:
            std = sde.marginal_prob(jnp.asarray([0.]), jnp.asarray([t]))[1]
            print(f't={t}, std={round(std, 5)}', file=sys.stderr)
        # raise Exception()

    print("Build one-step training and evaluation functions")
    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    lambda_optimize_fn = losses.lambda_optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    deterministic_latent_input = not hasattr(config.model,
                                             'deterministic_latent_input') or config.model.deterministic_latent_input
    latent_dim = -1 if not hasattr(config.model, 'latent_input_dim') else config.model.latent_input_dim
    single_t = None if not hasattr(config.model, 'single_t') else config.model.single_t
    train_clf = getattr(config.training, 'train_clf', False)
    train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting,
                                       lambda_z=config.model.lambda_z if hasattr(config.model, 'lambda_z') else None,
                                       lambda_model=lambda_model, x0_input=x0_input,
                                       deterministic_latent_input=deterministic_latent_input, latent_dim=latent_dim,
                                       single_t=single_t, config=config)
    # Pmap (and jit-compile) multiple training steps together for faster running
    p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
    lambda_train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimize_fn=lambda_optimize_fn,
                                              reduce_mean=reduce_mean, continuous=continuous,
                                              likelihood_weighting=likelihood_weighting,
                                              lambda_z=config.model.lambda_z if hasattr(config.model,
                                                                                        'lambda_z') else None,
                                              lambda_model=lambda_model, adversarial=True, x0_input=x0_input,
                                              config=config)
    # Pmap (and jit-compile) multiple training steps together for faster running
    lambda_p_train_step = jax.pmap(functools.partial(jax.lax.scan, lambda_train_step_fn), axis_name='batch',
                                   donate_argnums=1)
    eval_step_fn = losses.get_step_fn(sde, score_model, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting,
                                      lambda_z=config.model.lambda_z if hasattr(config.model, 'lambda_z') else None,
                                      x0_input=x0_input, deterministic_latent_input=deterministic_latent_input,
                                      latent_dim=latent_dim, single_t=single_t, config=config)
    # Pmap (and jit-compile) multiple evaluation steps together for faster running
    p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

    latent_step = losses.get_step_fn_latent(sde, score_model, lambda_model=lambda_model, x0_input=x0_input,
                                            deterministic_latent_input=deterministic_latent_input,
                                            latent_dim=latent_dim, config=config)
    # Pmap (and jit-compile) multiple evaluation steps together for faster execution
    p_latent_step = jax.pmap(functools.partial(jax.lax.scan, latent_step), axis_name='batch', donate_argnums=1)

    # Building sampling functions
    if config.training.snapshot_sampling and not x0_input:
        sampling_shape = (config.training.batch_size // jax.local_device_count(), config.data.image_size,
                          config.data.image_size, config.data.num_channels)
        sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

    # Replicate the training state to run on multiple devices
    pstate = flax_utils.replicate(state)
    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    if jax.host_id() == 0:
        logging.info("Starting training loop at step %d." % (initial_step,))
    rng = jax.random.fold_in(rng, jax.host_id())

    # JIT multiple training steps together for faster training
    n_jitted_steps = config.training.n_jitted_steps
    # Must be divisible by the number of steps jitted together
    assert config.training.log_freq % n_jitted_steps == 0 and \
           config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
           config.training.eval_freq % n_jitted_steps == 0 and \
           config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

    if train_clf:
        initial_step = 0
        num_train_steps = 20001

    clf_accs = {}
    total_labelled_samples = 0
    last_lambda_loss = 0
    if initial_step > 0 and lambda_model is not None:
        for adv_step in range(1000):
            # do multiple training steps only for lambda model
            batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))
            batch['label'] = inverse_scaler(batch['label'])
            rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
            next_rng = jnp.asarray(next_rng)
            (_, pstate), (ploss, pgrad) = lambda_p_train_step((next_rng, pstate), batch)
            if adv_step % 100 == 0:
                state = flax_utils.unreplicate(pstate)
                print(state.lambda_optimizer.target, file=sys.stderr)

    for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
        endsWithZeros = lambda x: str(x).endswith('0' * (len(str(x)) - 1))
        # logging.info("Training loop at step %d." % (step,))
        if endsWithZeros(step) or step % 10000 == 0:
            logging.info("Training loop at step %d." % (step,))
            # for param_name in ['sde_beta_min', 'sde_beta_max', 'sde_loss_weight', 'sde_beta']:
            #    if param_name in pstate.params_ema:
            #        print((param_name, pstate.params_ema[param_name]), file=sys.stderr)
            # print(f'Step {step}', file=sys.stderr)
            # print(f'Step {step}')
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))
        batch['label'] = inverse_scaler(batch['label'])
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)

        if getattr(config.data, 'debug_data', False):
            label = batch['label']
            idx = label >= 0
            if idx.sum() > 0:
                total_labelled_samples += idx.sum()
                print(("step", step, "labelled_samples:", idx.sum(), "total:", total_labelled_samples), file=sys.stderr)
            if step > 50000 // 64:
                print(("Finished", ' '), file=sys.stderr)
                return
            continue
            raise Exception()

        # Perform sanity check that data maps to single gaussian distribution
        if False and (step == initial_step or (endsWithZeros(step) and str(step)[0] == '1')):
            batch['forced_t'] = batch['label'] * 0. + 1.
            _, p_latent_result = p_latent_step((next_rng, pstate), batch)
            latent_result = flax.jax_utils.unreplicate(p_latent_result)
            abs_mean_err = np.absolute(latent_result['mean'].reshape(-1)).mean()
            abs_std_err = np.absolute(latent_result['std'].reshape(-1) - 1).mean()
            print("---- SDE Sanity Check ----", file=sys.stderr)
            print(("abs_mean_err:", abs_mean_err), file=sys.stderr)
            print(("abs_std_err:", abs_std_err), file=sys.stderr)
            if max(abs_mean_err * 1e-2, abs_std_err) > 1e-4:
                print("---- WARNING ---- SDE NOISE LEVEL NOT SUFFICIENT TO MAP TO PRIOR", file=sys.stderr)
                # raise Exception('SDE noise level not sufficient to map to noise')

        # Execute one training step
        is_lambda_reconstr_balanced_step = (lambda_reconstr_rate > 0. and (step % 100) == 0 and step > 0) or (
                lambda_method == 'lossprop' and (step % 100) == 0 and step > 0)

        if is_lambda_reconstr_balanced_step:
            # update lambda_reconstr_balanced
            n_ts = 20
            dsm_losses = []
            reconstr_losses = []
            ts = np.linspace(0. + 0.5 / n_ts, 1. - 0.5 / n_ts, n_ts)
            for t in ts:
                batch['forced_t'] = batch['label'] * 0. + t
                _, p_latent_result = p_latent_step((next_rng, pstate), batch)
                latent_result = flax.jax_utils.unreplicate(p_latent_result)
                dsm_losses.append(latent_result['dsm_losses'].mean())
                reconstr_losses.append(latent_result['reconstr_losses'].mean())
            # del batch['forced_t']
            dsm_losses = np.array(dsm_losses)
            if lambda_method == 'lossprop':
                lambda_reconstr_balanced = jnp.asarray(dsm_losses / dsm_losses.mean())
                state = flax_utils.unreplicate(pstate)
                state = state.replace(lambda_reconstr_balanced=lambda_reconstr_balanced)
                pstate = flax_utils.replicate(state)
                if (step % 10000) == 0:
                    np.save(os.path.join(lambda_reconstr_balanced_dir, 'loss_%7d' % step),
                            {'dsm_losses': dsm_losses})
                    fig = plt.figure()
                    plt.yscale('log')
                    plt.plot(dsm_losses, label='dsm_loss')
                    plt.legend()
                    plt.savefig(os.path.join(lambda_reconstr_balanced_dir, 'losses_%7d.png' % step),
                                bbox_inches='tight')
                    plt.close()
            else:
                reconstr_losses = np.array(reconstr_losses)
                reconstr_loss_ratios = dsm_losses / reconstr_losses
                if (step % 1000) == 0:
                    logging.info(
                        ('old lambda_reconstr_balanced', lambda_reconstr_balanced, lambda_reconstr_balanced.shape))
                lambda_reconstr_balanced = lambda_reconstr_rate * jnp.asarray(reconstr_loss_ratios).reshape(
                    lambda_reconstr_balanced.shape)
                state = flax_utils.unreplicate(pstate)
                state = state.replace(lambda_reconstr_balanced=lambda_reconstr_balanced)
                pstate = flax_utils.replicate(state)
                # store / plot information
                if (step % 1000) == 0:
                    logging.info(
                        ('new lambda_reconstr_balanced', lambda_reconstr_balanced, lambda_reconstr_balanced.shape))
                if (step % 1000) == 0:
                    np.save(os.path.join(lambda_reconstr_balanced_dir, 'ratios_%7d' % step),
                            {'reconstr_loss_ratios': reconstr_loss_ratios, 'rate': lambda_reconstr_rate,
                             'reconstr_losses': reconstr_losses, 'dsm_losses': dsm_losses})
                    fig = plt.figure()
                    plt.yscale('log')
                    plt.plot(dsm_losses, label='dsm_loss')
                    plt.plot(reconstr_losses, label='reconstr_loss')
                    plt.legend()
                    plt.savefig(os.path.join(lambda_reconstr_balanced_dir, 'losses_%7d.png' % step),
                                bbox_inches='tight')
                    plt.close()
                    fig = plt.figure()
                    plt.yscale('log')
                    plt.plot(lambda_reconstr_balanced)
                    plt.savefig(os.path.join(lambda_reconstr_balanced_dir, 'lambda_reconstr_balanced_%7d.png' % step),
                                bbox_inches='tight')
                    plt.close()

        if True:
            # logging.info(('lambda_reconstr_balanced', lambda_reconstr_balanced))
            (_, pstate), (ploss, pgrad) = p_train_step((next_rng, pstate), batch)
            loss = flax.jax_utils.unreplicate(ploss).mean()
            if False and pgrad is not None:
                if endsWithZeros(step):
                    grad = flax.jax_utils.unreplicate(pgrad)
                    grad_summary = {}
                    for k, g in grad.items():
                        if 'kernel' not in g:
                            continue
                        grad_summary[k] = {'absmean': np.absolute(g['kernel']).mean()}
                    grad_summary['train_loss'] = {'absmean': loss}
                    pickle.dump(grad_summary, open(f'info/grad_{step}.p', "wb"))

            # Log to console, file and tensorboard on host 0
            if jax.host_id() == 0 and (endsWithZeros(step) or step % 1000 == 0 or (
                    ((step % 10) == 0) and step <= 600100 and step >= 600000)):
                # if jax.host_id() == 0 and step % config.training.log_freq == 0:
                logging.info("step: %d, training_loss: %.5e" % (step, loss))
                writer.scalar("training_loss", loss, step)
                if lambda_reconstr_balanced is not None:
                    logging.info(('lambda_reconstr_balanced', lambda_reconstr_balanced))

                if 'forced_t' in batch:
                    del batch['forced_t']

                if lambda_model is not None and ((step % 1000) == 0):
                    logging.info(f'eval lambda model')
                    # for param_name in ['t_0']:
                    #    if param_name in pstate.lambda_params_ema:
                    #        print((param_name, pstate.lambda_params_ema[param_name]), file=sys.stderr)
                    #        print((param_name, 1. / (1. + np.exp(-pstate.lambda_params_ema[param_name]))),
                    #              file=sys.stderr)
                    forced_t = jnp.asarray(
                        batch['label'].shape[0] * list(range(int(np.prod(batch['label'].shape[1:]))))).astype(float)
                    forced_t = forced_t / forced_t.max()
                    batch['forced_t'] = forced_t.reshape(batch['label'].shape)
                    _, p_latent_result = p_latent_step((next_rng, pstate), batch)
                    latent_result = flax.jax_utils.unreplicate(p_latent_result)
                    del batch['forced_t']

                    np.save(os.path.join(lambda_reconstr_balanced_dir, 'lm_%7d' % step),
                            {'lm_params': pstate.lambda_params_ema, 't': latent_result['t'],
                             'lambda_xt': latent_result['lambda_xt'], 'forced_t_ret': latent_result['forced_t'],
                             'forced_t_in': forced_t})
                    fig = plt.figure()
                    plt.yscale('log')
                    plt.plot(latent_result['t'].reshape(-1), latent_result['lambda_xt'].reshape(-1))
                    plt.savefig(os.path.join(lambda_reconstr_balanced_dir, 'lambda_xt_%7d.png' % step),
                                bbox_inches='tight')
                    plt.close()

                    # for param_name in pstate.lambda_params_ema:
                    #    print((param_name, np.exp(pstate.lambda_params_ema[param_name]) if 'std' in param_name else
                    #    pstate.lambda_params_ema[param_name]), file=sys.stderr)
                    # raise Exception()

        is_lambda_step = (((step // n_jitted_steps) % 1) == 0) and lambda_model is not None
        if is_lambda_step:
            # print(("Lambda train step!",), file=sys.stderr)
            for _ in range(1):
                (_, pstate), (ploss, pgrad) = lambda_p_train_step((next_rng, pstate), batch)
            last_lambda_loss = flax.jax_utils.unreplicate(ploss).mean()
            # print(("ploss:", ploss), file=sys.stderr)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if not train_clf and (
                step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0):
            saved_state = flax_utils.unreplicate(pstate)
            saved_state = saved_state.replace(rng=rng)
            checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                        step=step // config.training.snapshot_freq_for_preemption,
                                        keep=1)

        # Report the loss on an evaluation dataset periodically
        # if step % config.training.eval_freq == 0:
            logging.info(f'eval at step {step}')
        if not train_clf:
            if endsWithZeros(step) or step % 10000 == 0:
                eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))
                eval_batch['label'] = inverse_scaler(eval_batch['label'])
                rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
                next_rng = jnp.asarray(next_rng)
                (_, _), (peval_loss, _) = p_eval_step((next_rng, pstate), eval_batch)
                eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
                if jax.host_id() == 0:
                    logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
                    writer.scalar("eval_loss", eval_loss, step)
        else:
            if step % 1000 == 0:
                total_eval_loss = 0.
                for i_eval_batch, eval_batch_tf in enumerate(eval_ds):
                    eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), eval_batch_tf)
                    eval_batch['label'] = inverse_scaler(eval_batch['label'])
                    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
                    next_rng = jnp.asarray(next_rng)
                    (_, _), (peval_loss, _) = p_eval_step((next_rng, pstate), eval_batch)
                    eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
                    total_eval_loss += eval_loss
                    if i_eval_batch * config.eval.batch_size >= 10000:
                        acc = total_eval_loss / (i_eval_batch+1) * 100
                        clf_accs[step] = acc
                        np.save(os.path.join(lambda_reconstr_balanced_dir, 'clf_accs'), clf_accs)
                        logging.info(f'step: {step}, eval accuracy: {acc}')
                        break

        # Save a checkpoint periodically and generate samples if needed
        if not train_clf and (step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps):
            # Save the checkpoint.
            if jax.host_id() == 0:
                saved_state = flax_utils.unreplicate(pstate)
                saved_state = saved_state.replace(rng=rng)
                checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                            step=step // config.training.snapshot_freq,
                                            keep=np.inf)

            # Generate and save samples
            if config.training.snapshot_sampling and not x0_input:
                logging.info("step: %d, sample snapshots" % (step,))
                rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
                sample_rng = jnp.asarray(sample_rng)
                sample, n = sampling_fn(sample_rng, pstate)
                this_sample_dir = os.path.join(
                    sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
                tf.io.gfile.makedirs(this_sample_dir)
                image_grid = sample.reshape((-1, *sample.shape[2:]))
                nrow = int(np.sqrt(image_grid.shape[0]))
                sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                    np.save(fout, sample)

                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    utils.save_image(image_grid, fout, nrow=nrow, padding=2)


def evaluate(config,
             workdir,
             eval_folder="eval"):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    import os
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if hasattr(config.training, 'experiment_name'):
        eval_dir += f'_{config.training.experiment_name}'
    if hasattr(config.eval, 'eval_dir_postfix'):
        eval_dir += f'{config.eval.eval_dir_postfix}'
    tf.io.gfile.makedirs(eval_dir)
    latent_dir = os.path.join(eval_dir, 'latent')
    tf.io.gfile.makedirs(latent_dir)

    rng = jax.random.PRNGKey(config.seed + 1)

    # Build data pipeline
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=1,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    """
    rng, model_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
    optimizer = losses.get_optimizer(config).create(initial_params)
    state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                         model_state=init_model_state,
                         ema_rate=config.model.ema_rate,
                         params_ema=initial_params,
                         rng=rng)  # pytype: disable=wrong-keyword-args
    """
    x0_input = hasattr(config.model, 'include_latent_input') and config.model.include_latent_input
    deterministic_latent_input = not hasattr(config.model,
                                             'deterministic_latent_input') or config.model.deterministic_latent_input
    latent_dim = -1 if not hasattr(config.model, 'latent_input_dim') else config.model.latent_input_dim

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    if hasattr(config.training, 'experiment_name'):
        checkpoint_dir += f'_{config.training.experiment_name}'

    rng, step_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(step_rng, config, x0_input=x0_input)

    use_vae_samples = hasattr(config.eval, 'use_vae_samples') and config.eval.use_vae_samples
    vae_sample_t = config.eval.vae_sample_t if hasattr(config.eval, 'vae_sample_t') else None
    vae_sample_is_p0 = hasattr(config.eval, 'vae_sample_is_p0') and config.eval.vae_sample_is_p0

    frozen_encoder = hasattr(config.model, 'frozen_encoder') and config.model.frozen_encoder
    focus = None
    if frozen_encoder:
        # focus = flax.optim.ModelParamTraversal(lambda path, _: 'Encoder_0' not in path)
        focus = flax.optim.ModelParamTraversal(shouldBeOptimized)
    optimizer = losses.get_optimizer(config).create(initial_params, focus=focus)
    # optimizer = losses.get_optimizer(config).create(initial_params)

    rng, step_rng = jax.random.split(rng)
    lambda_model, lambda_init_model_state, lambda_initial_params, lambda_optimizer = None, None, None, None
    use_lambda_model = hasattr(config.model, 'include_lambda_model') and config.model.include_lambda_model
    if use_lambda_model:
        lambda_model, lambda_init_model_state, lambda_initial_params = mutils.init_lambda_model(step_rng, config)
        lambda_optimizer = losses.get_optimizer(config).create(lambda_initial_params)

    lambda_reconstr_rate = getattr(config.model, 'lambda_reconstr_rate', 0.0)
    lambda_reconstr_balanced = None
    if lambda_reconstr_rate > 0:
        lambda_reconstr_balanced = jnp.asarray([1.0] * 20)

    start_checkpoint_dir = getattr(config.training, 'start_checkpoint_dir', '')
    state_new = mutils.State(step=0,
                             optimizer=optimizer,
                             lr=config.optim.lr,
                             model_state=init_model_state,
                             ema_rate=config.model.ema_rate,
                             params_ema=initial_params,
                             lambda_step=0,
                             lambda_optimizer=lambda_optimizer,
                             lambda_lr=1e-3,
                             lambda_model_state=lambda_init_model_state,
                             lambda_ema_rate=config.model.ema_rate,
                             lambda_params_ema=lambda_initial_params,
                             lambda_reconstr_balanced=lambda_reconstr_balanced,
                             rng=rng)  # pytype: disable=wrong-keyword-args

    old_state = 'pretrained' in config.training.experiment_name  # or 'pretrained' in start_checkpoint_dir  # config.data.image_size > 1
    if old_state:
        state = mutils.Stateold(step=0,
                                optimizer=optimizer,
                                lr=config.optim.lr,
                                model_state=init_model_state,
                                ema_rate=config.model.ema_rate,
                                params_ema=initial_params,
                                rng=rng)  # pytype: disable=wrong-keyword-args
    else:
        state = state_new
    """
    old_state = True  # config.data.image_size > 1
    if old_state:
        state = mutils.Stateold(step=0,
                                optimizer=optimizer,
                                lr=config.optim.lr,
                                model_state=init_model_state,
                                ema_rate=config.model.ema_rate,
                                params_ema=initial_params,
                                rng=rng)  # pytype: disable=wrong-keyword-args
        ckpt = config.eval.begin_ckpt
        # state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
        try:
            state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
        except:
            old_state = False
    if not old_state:
        state = mutils.State(step=0,
                             optimizer=optimizer,
                             lr=config.optim.lr,
                             model_state=init_model_state,
                             ema_rate=config.model.ema_rate,
                             params_ema=initial_params,
                             lambda_step=0,
                             lambda_optimizer=lambda_optimizer,
                             lambda_lr=1e-3,
                             lambda_model_state=lambda_init_model_state,
                             lambda_ema_rate=config.model.ema_rate,
                             lambda_params_ema=lambda_initial_params,
                             rng=rng)  # pytype: disable=wrong-keyword-args
     """

    rng, step_rng = jax.random.split(rng)
    params_clf = mnist_clf.get_initial_params(step_rng)
    params_clf = checkpoints.restore_checkpoint('ckpt_mnist_classifier', params_clf, step=10)
    predict_labels = lambda image_batch: mnist_clf.CNN().apply({'params': params_clf}, scaler(image_batch))

    # Setup SDEs
    sdeN = getattr(config.eval, 'n_sampling_steps', config.model.num_scales)
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=sdeN)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=sdeN)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=sdeN)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, score_model,
                                       train=False, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean,
                                       continuous=continuous, likelihood_weighting=likelihood_weighting,
                                       lambda_model=lambda_model, x0_input=x0_input, config=config)
        # Pmap (and jit-compile) multiple evaluation steps together for faster execution
        p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step), axis_name='batch', donate_argnums=1)

    if hasattr(config.eval, 'enable_latent') and config.eval.enable_latent:
        latent_step = losses.get_step_fn_latent(sde, score_model, lambda_model=lambda_model, x0_input=x0_input,
                                                deterministic_latent_input=deterministic_latent_input,
                                                latent_dim=latent_dim, config=config)
        # Pmap (and jit-compile) multiple evaluation steps together for faster execution
        p_latent_step = jax.pmap(functools.partial(jax.lax.scan, latent_step), axis_name='batch', donate_argnums=1)

    enable_latentgrid = hasattr(config.eval, 'enable_latentgrid') and config.eval.enable_latentgrid
    if enable_latentgrid:
        latentgrid_step = losses.get_step_fn_latent(sde, score_model, lambda_model=lambda_model, x0_input=x0_input,
                                                    latentscorefn=True,
                                                    deterministic_latent_input=deterministic_latent_input,
                                                    latent_dim=latent_dim, config=config)
        # Pmap (and jit-compile) multiple evaluation steps together for faster execution
        p_latentgrid_step = jax.pmap(functools.partial(jax.lax.scan, latentgrid_step), axis_name='batch',
                                     donate_argnums=1)

    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        additional_dim=None,
                                                        uniform_dequantization=True, evaluation=True)
    if config.eval.bpd_dataset.lower() == 'train':
        ds_bpd = train_ds_bpd
        bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds_bpd = eval_ds_bpd
        bpd_num_repeats = 1  # 5
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    # TODO: Set apply denoising or apply ode grid
    apply_denoising = False and config.data.image_size > 1  # True
    apply_ode_grid = False and config.data.image_size <= 1
    denoise_samples_wrong_z = hasattr(config.eval,
                                      'denoise_samples_wrong_z') and config.eval.denoise_samples_wrong_z
    eval_intermediate_sampling_steps = getattr(config.eval, 'eval_intermediate_sampling_steps', False)

    print(f'apply_denoising={apply_denoising}, apply_ode_grid={apply_ode_grid}', file=sys.stderr)

    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd or apply_denoising:
        likelihood_fn = likelihood.get_likelihood_fn(sde, score_model, inverse_scaler)

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling or denoise_samples_wrong_z:
        sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                          config.data.image_size, config.data.image_size,
                          config.data.num_channels)
        conditional_sampling = getattr(config.eval, 'conditional_sampling', False)
        if conditional_sampling:
            # load classifier
            rng, step_rng = jax.random.split(rng)
            logging.info('load classifier')
            classifier, classifier_params = mutils.create_classifier(step_rng, config.eval.batch_size,
                                                                     'wideresnet_noise_conditional/')
            logging.info('load classifier finished')
            sampling_fn = controllable_generation.get_conditional_sampling_fn(config, sde, score_model, sampling_shape,
                                                                              inverse_scaler, sampling_eps,
                                                                              probability_flow=config.sampling.probability_flow,
                                                                              classifier=classifier,
                                                                              classifier_params=classifier_params)
        else:
            sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler,
                                                   sampling_eps,
                                                   probability_flow=config.sampling.probability_flow)

    if eval_intermediate_sampling_steps:
        sampling_shape = (2,
                          config.data.image_size, config.data.image_size,
                          config.data.num_channels)
        sampling_fn_intermediate = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler,
                                                            sampling_eps,
                                                            probability_flow=config.sampling.probability_flow,
                                                            return_intermediate=True)

    if apply_denoising or apply_ode_grid or denoise_samples_wrong_z:
        sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                          config.data.image_size, config.data.image_size,
                          config.data.num_channels)
        sampling_fn_sde = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler,
                                                   sampling_eps, probability_flow=False)
        sampling_fn_ode = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler,
                                                   sampling_eps, probability_flow=True)

    # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
    rng = jax.random.fold_in(rng, jax.host_id())

    # A data class for storing intermediate results to resume evaluation after pre-emption
    @flax.struct.dataclass
    class EvalMeta:
        ckpt_id: int
        sampling_round_id: int
        bpd_round_id: int
        rng: Any

    # Add one additional round to get the exact number of samples as required.
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
    if config.eval.enable_bpd:
        num_bpd_rounds = len(ds_bpd) * bpd_num_repeats
    else:
        num_bpd_rounds = 2000 * bpd_num_repeats

    # Restore evaluation after pre-emption
    eval_meta = EvalMeta(ckpt_id=config.eval.begin_ckpt, sampling_round_id=-1, bpd_round_id=-1, rng=rng)
    eval_meta = checkpoints.restore_checkpoint(
        eval_dir, eval_meta, step=None, prefix=f"meta_{jax.host_id()}_")

    if eval_meta.bpd_round_id < num_bpd_rounds - 1:
        begin_ckpt = eval_meta.ckpt_id
        begin_bpd_round = eval_meta.bpd_round_id + 1
        begin_sampling_round = 0

    elif eval_meta.sampling_round_id < num_sampling_rounds - 1:
        begin_ckpt = eval_meta.ckpt_id
        begin_bpd_round = num_bpd_rounds
        begin_sampling_round = eval_meta.sampling_round_id + 1

    else:
        begin_ckpt = eval_meta.ckpt_id + 1
        begin_bpd_round = 0
        begin_sampling_round = 0

    rng = eval_meta.rng

    # Use inceptionV3 for images with resolution higher than 256.
    eval_samples = not hasattr(config.eval, "eval_samples") or config.eval.eval_samples
    if eval_samples:
        inceptionv3 = config.data.image_size >= 256
        inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    ckpts = list(range(begin_ckpt, config.eval.end_ckpt + 1))
    # ckpts = [begin_ckpt, config.eval.end_ckpt]
    for ckpt in ckpts:
        print(("ckpt:", ckpt), file=sys.stderr)
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
        while not tf.io.gfile.exists(ckpt_filename):
            if not waiting_message_printed and jax.host_id() == 0:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                logging.warning(f'because this file not exists: {ckpt_filename}')
                waiting_message_printed = True
            time.sleep(60)

        print(("found checkpoint file",), file=sys.stderr)
        # Wait for 2 additional mins in case the file exists but is not ready for reading
        state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
        """
        try:
            state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
        except:
            print(("exception restoring checkpoint",), file=sys.stderr)
            time.sleep(60)
            try:
                state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
            except:
                print(("again exception restoring checkpoint",), file=sys.stderr)
                time.sleep(120)
                state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
        """

        print(("restored checkpoint",), file=sys.stderr)
        save_checkpoints = False

        # Replicate the training state for executing on multiple devices
        pstate = flax.jax_utils.replicate(state)
        # print(("state", type(state)), file=sys.stderr)
        # print(("state", state), file=sys.stderr)

        if apply_denoising:
            this_denoising_dir = 'denoise_samples_clipped_05'
            tf.io.gfile.makedirs(this_denoising_dir)

            ds_iter = iter(ds_bpd)

            reconstruction_error = Counter()
            n_batches = 2
            n_per_batch = 4
            stds = [0.3, 0.5, .7]  # [0., 0.1, 0.3, 0.5]
            saltpepperps = [0.05, 0.2, 0.5]
            init_ts = [
                0.5]  # [0.8 ** i for i in range(1, 6)]  # [0., 0.001, 0.003, 0.01, 0.03, 0.1]  # [0.5 ** i for i in range(1, 2)]
            sde_stds = []
            # sde_stds = [0., 0.00010991, 0.001993, 0.1037178] + [0.] * 10
            batch = next(ds_iter)
            eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
            eval_batch['label'] = inverse_scaler(eval_batch['label'])
            if len(sde_stds) == 0:
                for init_t in init_ts:
                    sde_std = \
                        sde.marginal_prob(eval_batch['image'], jnp.asarray([init_t]),
                                          params=pstate.params_ema)[1]
                    print(("init_t, std:", init_t, sde_std), file=sys.stderr)
                    sde_stds.append(sde_std)
            for i_batch in range(n_batches):
                logging.info(f'Start denoising of batch {i_batch}')
                denoised_images = []
                for i_inner_batch in range(n_per_batch):
                    logging.info(f'Start denoising of sample {i_inner_batch}')
                    rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
                    step_rng = jnp.asarray(step_rng)
                    batch = next(ds_iter)
                    eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)

                    only_likelihood = False
                    if only_likelihood:
                        logging.info(f'only_likelihood')
                        for i, init_t in enumerate(init_ts):
                            logging.info(f'init_t={init_t}, std={sde_stds[i]}')
                            bpd, z, _ = likelihood_fn(step_rng, pstate, eval_batch['image'], t_data=init_t)
                            logging.info(f'bpd={jnp.mean(bpd)}')
                        return

                    noise_model = 'gauss' if i_batch % 2 == 0 else 'saltpepper'
                    noise_levels = stds if noise_model == 'gauss' else saltpepperps
                    for noise_level in noise_levels:
                        print(("noise_level:", noise_level), file=sys.stderr)
                        clean_img = inverse_scaler(eval_batch['image'])
                        if noise_model == 'gauss':
                            noisy_image = (clean_img + jax.random.normal(rng, eval_batch['image'].shape) * noise_level)
                        else:
                            noisy_image = (clean_img + jax.random.choice(rng, jnp.asarray([-1., 0., 1.]),
                                                                         shape=eval_batch['image'].shape, replace=True,
                                                                         p=jnp.asarray(
                                                                             [noise_level * 0.5, 1. - noise_level,
                                                                              noise_level * 0.5])))
                        noisy_image = jnp.asarray(
                            np.float32(np.clip(noisy_image * 255., 0, 255).astype(np.uint8)) / 255.)

                        # print(("noisy_image.shape:", noisy_image.shape), file=sys.stderr)
                        # blur_cv2 = cv2.GaussianBlur(np.float32(np.array(noisy_image)).reshape((28, 28, 1)), (5, 5), 0)
                        # blur_cv2 = cv2.GaussianBlur(np.array(noisy_image).reshape((28, 28, 1)), (5, 5), 0)
                        # blur_cv2 = cv2.GaussianBlur(np.array(noisy_image).reshape(28, 28), (5, 5), 0)
                        # blur_cv2 = cv2.medianBlur(blur_cv2, 5)

                        denoised_images.append(clean_img)
                        denoised_images.append(noisy_image)
                        err = jnp.square(clean_img - noisy_image).mean()
                        reconstruction_error[f'noisy_{noise_level}'] += err
                        print(("err noise:", err), file=sys.stderr)

                        # Model denoising using SDE
                        # bpds = []
                        # for init_t in init_ts:
                        #   bpd, z, _ = likelihood_fn(step_rng, pstate, scaler(noisy_image), t_data=init_t)
                        #   bpds.append(bpd)
                        # bpds = jnp.concatenate(bpds, axis=0)
                        bpds = jnp.asarray([0.])
                        print(("bpds:", bpds), file=sys.stderr)
                        i_mle = bpds.reshape(-1).argmin()
                        print(("i_mle:", i_mle), file=sys.stderr)
                        t_mle = init_ts[i_mle]
                        # jnp.mean(jnp.asarray(bpds))
                        # print("bpds:", file=sys.stderr)
                        # print(bpds, file=sys.stderr)
                        print(("t_mle:", t_mle), file=sys.stderr)
                        print(("std, std_mle:", noise_level, sde_stds[i_mle]), file=sys.stderr)
                        sample, _ = sampling_fn_sde(step_rng, pstate, prior_seed=scaler(noisy_image[0]),
                                                    init_t=jnp.asarray(t_mle)[None])  #
                        denoised_images.append(sample)
                        err = jnp.square(clean_img - denoised_images[-1]).mean()
                        reconstruction_error[f'sampleSDE_{noise_level}'] += err
                        print(("err sampleSDE:", err), file=sys.stderr)

                        sample, _ = sampling_fn_ode(step_rng, pstate, prior_seed=scaler(noisy_image[0]),
                                                    init_t=jnp.asarray(t_mle)[None])  #
                        denoised_images.append(sample)
                        err = jnp.square(clean_img - denoised_images[-1]).mean()
                        reconstruction_error[f'sampleODE_{noise_level}'] += err
                        print(("sampleMinMax:", sample.min(), sample.max()), file=sys.stderr)

                        print(("err sampleODE:", err), file=sys.stderr)

                        # CV2 dendoising
                        blur_cv2 = cv2.GaussianBlur(np.array(noisy_image).reshape(28, 28), (3, 3), 0)
                        blur_cv2 = cv2.medianBlur(blur_cv2, 5)
                        denoised_images.append(blur_cv2[None, None, :, :, None])
                        err = jnp.square(clean_img - denoised_images[-1]).mean()
                        reconstruction_error[f'blur_cv2_{noise_level}'] += err
                        print(("err blur_cv2:", err), file=sys.stderr)

                        img_bilateral = cv2.medianBlur(np.array(noisy_image).reshape(28, 28), 5)
                        denoised_images.append(img_bilateral[None, None, :, :, None])
                        err = jnp.square(clean_img - denoised_images[-1]).mean()
                        reconstruction_error[f'bilateral_{noise_level}'] += err
                        print(("err bilateral:", err), file=sys.stderr)

                        # img_nlmeans = np.float32(cv2.fastNlMeansDenoising(np.array(noisy_image * 255.).reshape(28, 28).astype(np.uint8))) / 255.
                        # denoised_images.append(img_nlmeans[None, None, :, :, None])
                        # print(("err nlmeans:", jnp.square(clean_img - denoised_images[-1]).mean()), file=sys.stderr)

                img_images = jnp.concatenate(denoised_images, axis=0).clip(a_min=0., a_max=1.)
                image_grid = img_images.reshape((-1, *img_images.shape[2:]))
                nrow = int(len(denoised_images) / (n_per_batch * len(stds)))  # int(np.sqrt(image_grid.shape[0]))
                print(("image_grid.shape", image_grid.shape), file=sys.stderr)
                # image_grid = np.clip(image_grid * 255., 0, 255).astype(np.uint8)
                with tf.io.gfile.GFile(
                        os.path.join(this_denoising_dir, f'denoise_{i_batch}.png'), "wb") as fout:
                    utils.save_image(image_grid, fout, nrow=nrow, padding=2)

                print(reconstruction_error, file=sys.stderr)

            for key, val in reconstruction_error.items():
                val /= n_batches * n_per_batch
                print((f'err {key}:', val), file=sys.stderr)
                reconstruction_error[key] = val

            print(reconstruction_error, file=sys.stderr)
            return

        if eval_intermediate_sampling_steps:
            logging.info(f'sampling_fn_intermediate')
            rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
            step_rng = jnp.asarray(step_rng)
            xs, xs_mean = sampling_fn_intermediate(step_rng, pstate)
            np.save(os.path.join(eval_dir, f'intermediate_sample_steps'), {'xs': xs, 'xs_mean': xs_mean})
            logging.info(f'exit program after sampling_fn_intermediate')
            return

        if apply_ode_grid:
            logging.info(f'apply_ode_grid')

            rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
            step_rng = jnp.asarray(step_rng)

            mu = 0.
            sigma = 1.
            dist = sps.norm(loc=mu, scale=sigma)

            n_z0_vals = 25
            # z0_percentiles = np.linspace(0., 1., n_z0_vals + 2)
            # z0_vals = np.array([dist.ppf(ppf) for ppf in z0_percentiles[1:-1]])
            z0_vals = np.linspace(-1., 1., n_z0_vals)
            logging.info(f'z0_vals={z0_vals}')

            sparse_z0 = config.data.num_channels > 2
            if sparse_z0:
                z0 = np.zeros((n_z0_vals * config.data.num_channels, config.data.num_channels))
                for i1 in range(n_z0_vals):
                    for i2 in range(config.data.num_channels):
                        z0[i1 * config.data.num_channels + i2, i2] = z0_vals[i1]
            else:
                z0 = np.zeros((n_z0_vals ** 2, 2))
                for i1 in range(n_z0_vals):
                    for i2 in range(n_z0_vals):
                        z0[i1 * n_z0_vals + i2, 0] = z0_vals[i1]
                        z0[i1 * n_z0_vals + i2, 1] = z0_vals[i2]
            z0 = jnp.asarray(z0)
            z = z0[:, None, None, :]

            logging.info(f'sampling_fn_ode')
            logging.info(f'z0_vals.shape={z0_vals.shape}')
            logging.info(f'z.shape={z.shape}')
            sample, _ = sampling_fn_ode(step_rng, pstate, prior_seed=z)  # , init_t=jnp.asarray([1.]))
            logging.info(f'sampling_fn_ode Finished')
            np.save(f'ae_z_grid_transformed_{config.data.num_channels}', {'zT': z, 'z0': sample})
            print(("Finished",), file=sys.stderr)
            return

        for param_name in ['sde_beta_min', 'sde_beta_max', 'sde_loss_weight', 'sde_beta']:
            if param_name in pstate.params_ema:
                print((param_name, pstate.params_ema[param_name]), file=sys.stderr)

        if hasattr(config.eval, 'enable_latent') and config.eval.enable_latent:
            print(("start latent:",), file=sys.stderr)
            print(("enable latent:",), file=sys.stderr)
            store_batchwise = False
            store_latent = True

            for ds_str in ['eval', 'train']:  # , 'train'
                latent_map = {'z0': [], 'z0_mean': [], 'label': [], 'x0': [], 'loss': defaultdict(list),
                              'reg_loss': defaultdict(list), 'reconstr_loss': defaultdict(list)}
                logging.info(f'Start latent evaluation of {ds_str} dataset')
                latent = {}
                eval_iter = iter(eval_ds if ds_str == 'eval' else train_ds)  # pytype: disable=wrong-arg-types
                for i, batch in enumerate(eval_iter):
                    eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
                    eval_batch['label'] = inverse_scaler(eval_batch['label'])

                    z0 = None
                    t_vals = np.linspace(0., 1.,
                                         num=21)  # [0.]  # np.linspace(0., 1., num=11)  # if config.data.image_size > 1 else [0.]
                    for t in t_vals:  # [0., 1e-3, 1e-2, 1e-1, 0.2, 0.5]:  #
                        eval_batch['forced_t'] = eval_batch['label'] * 0. + t
                        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
                        next_rng = jnp.asarray(next_rng)
                        _, p_latent_result = p_latent_step((next_rng, pstate), eval_batch)
                        latent_result = flax.jax_utils.unreplicate(p_latent_result)
                        # z0 = jnp.asarray(latent_result['latent'])
                        # print(("z0.shape", z0.shape), file=sys.stderr)
                        # z0 = jnp.concatenate([z0[i] for i in range(z0.shape[0])], axis=0)
                        # print(("z0.shape", z0.shape), file=sys.stderr)
                        z0 = p_latent_result['latent'] if deterministic_latent_input else p_latent_result['z_mean']
                        if not deterministic_latent_input:
                            std = jnp.exp(0.5 * latent_result['z_logvar'])
                            z0_sample = latent_result['z_mean'] + std * jax.random.normal(rng, std.shape)
                            latent_map['z0'].append(z0_sample)
                            latent_map['z0_mean'].append(latent_result['z_mean'])
                            latent_map['label'].append(latent_result['label'])
                        elif x0_input:
                            latent_map['z0'].append(z0)
                            latent_map['label'].append(latent_result['label'])
                        if config.data.image_size <= 1:
                            latent_map['x0'].append(latent_result['data'])
                        latent_map['loss'][float(latent_result['t'].reshape(-1)[0])].append(latent_result['losses'])
                        latent_map['reg_loss'][float(latent_result['t'].reshape(-1)[0])].append(
                            latent_result['reg_losses'])
                        latent_map['reconstr_loss'][float(latent_result['t'].reshape(-1)[0])].append(
                            latent_result['reconstr_losses'])

                        for key, val in latent_result.items():
                            jnp_val = jnp.asarray(val if val is not None else [False])
                            if key not in latent:
                                latent[key] = {}
                            latent[key][t] = jnp_val if t not in latent[key] else jnp.concatenate(
                                (latent[key][t], jnp_val))
                        if i == 0:
                            # print((f'lambda mean for t={t}:', latent_result['lambda_xt'][0].mean()), file=sys.stderr)
                            """
                            i1 = latent_result['perturbed_data'][0].reshape(latent_result['perturbed_data'][0].shape[0],
                                                                            -1).round(3)
                            i2 = latent_result['t'][0].reshape(-1, 1).round(3)
                            i3 = latent_result['lambda_xt'][0].reshape(-1, 1)
                            i4 = latent_result['losses'][0].reshape(-1, 1)
                            some_info = np.concatenate((i1, i2, i3, i4), axis=-1)
                            print("Lambda info:", file=sys.stderr)
                            print(some_info[:5], file=sys.stderr)
                            """
                            pass
                        if i == 0 and enable_latentgrid:
                            z0s = [None]  # [-1.2, -0.8, 0., 2., 4.5, 5.]
                            coor_x = coor_y = np.linspace(-1.5, 1.5, 25)
                            xx, yy = np.meshgrid(coor_x, coor_y)
                            coor_grid = np.dstack([xx, yy]).reshape(-1, 2)
                            print(("eval_batch['image'].shape", eval_batch['image'].shape), file=sys.stderr)
                            coor_grid_orig = coor_grid
                            batch_size = eval_batch['image'].shape[2]
                            assert coor_grid.shape[0] <= batch_size
                            while coor_grid.shape[0] < batch_size:
                                coor_grid = jnp.concatenate(
                                    (coor_grid, coor_grid[:min(coor_grid.shape[0], batch_size - coor_grid.shape[0])]),
                                    axis=0)
                            print(("coor_grid.shape", coor_grid.shape), file=sys.stderr)
                            coor_grid = coor_grid.reshape(eval_batch['image'].shape)
                            print(("coor_grid.shape", coor_grid.shape), file=sys.stderr)
                            eval_batch['forced_image'] = coor_grid
                            rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
                            next_rng = jnp.asarray(next_rng)

                            if 'grid_score' not in latent:
                                latent['grid_score'] = {}
                            if 'grid_lambda_xt' not in latent:
                                latent['grid_lambda_xt'] = {}
                            for z0_val in z0s:
                                if z0_val is not None:
                                    eval_batch['z0'] = (eval_batch['label'] * 0. + z0_val).reshape(
                                        (*eval_batch['image'].shape[:-3], 1))
                                _, p_latentgrid_result = p_latentgrid_step((next_rng, pstate), eval_batch)
                                latentgrid_result = flax.jax_utils.unreplicate(p_latentgrid_result)
                                if not x0_input:
                                    score = jnp.asarray(latentgrid_result['score'])
                                    score = score.reshape((batch_size, -1))
                                if lambda_model is not None:
                                    lambda_xt = jnp.asarray(latentgrid_result['lambda_xt'])
                                    lambda_xt = lambda_xt.reshape((batch_size, -1))
                                    if lambda_xt.shape[0] > 1:
                                        lambda_xt = lambda_xt.reshape((batch_size, -1))
                                if z0_val is not None:
                                    key = z0_val
                                    if not x0_input:
                                        if key not in latent['grid_score']:
                                            latent['grid_score'][key] = {}
                                        latent['grid_score'][key][t] = score[:coor_grid_orig.shape[0]]
                                    if lambda_model is not None:
                                        if key not in latent['grid_lambda_xt']:
                                            latent['grid_lambda_xt'][key] = {}
                                        latent['grid_lambda_xt'][key][t] = lambda_xt[:coor_grid_orig.shape[0]]
                                else:
                                    if not x0_input:
                                        latent['grid_score'][t] = score[:coor_grid_orig.shape[0]]
                                    if lambda_model is not None:
                                        latent['grid_lambda_xt'][t] = lambda_xt[:coor_grid_orig.shape[0]]
                            latent['grid_xx'] = xx
                            latent['grid_yy'] = yy

                        # if i == 0:
                        #    print((f'2 t={t}'), file=sys.stderr)
                        if denoise_samples_wrong_z and i == 0:
                            print((f'3 t={t}'), file=sys.stderr)
                            # image shape: (1, 1, 100, 28, 28, 1)
                            # label shape: (1, 1, 100)
                            # latent image/label shape: (1, 100, ...)
                            denoise_labels = np.array([0., 1., 2., 3.]) * 2. - 1
                            n_dsamples = len(denoise_labels)
                            denoise_data = jnp.concatenate(
                                [latent['data'][t][0][latent['label'][t][0] == l][:1] for l in denoise_labels], axis=0)
                            denoise_perturbed_data = jnp.concatenate(
                                [latent['perturbed_data'][t][0][latent['label'][t][0] == l][:1] for l in
                                 denoise_labels], axis=0)
                            denoise_latent = jnp.concatenate(
                                [latent['latent'][t][0][latent['label'][t][0] == l][:1] for l in denoise_labels],
                                axis=0)
                            # denoise_perturbed_data = jnp.array(np.array(denoise_perturbed_data))
                            # denoise_latent = jnp.array(np.array(denoise_latent))
                            # raise Exception((latent['data'][t].shape, latent['perturbed_data'][t].shape, latent['label'][t].shape))

                            # create denoising samples
                            dgrid_perturbed_data = jnp.concatenate(
                                [denoise_perturbed_data[i // n_dsamples:i // n_dsamples + 1] for i in
                                 range(n_dsamples ** 2)], axis=0)
                            dgrid_latent = jnp.concatenate(
                                [denoise_latent[i % n_dsamples:i % n_dsamples + 1] for i in range(n_dsamples ** 2)],
                                axis=0)

                            # raise Exception((dgrid_perturbed_data.shape, dgrid_latent.shape))
                            prior_seed = dgrid_perturbed_data
                            z0input = dgrid_latent
                            batch_size = latent['data'][t][0].shape[0]
                            assert prior_seed.shape[0] <= batch_size
                            while prior_seed.shape[0] < batch_size:
                                prior_seed = jnp.concatenate(
                                    (prior_seed,
                                     prior_seed[:min(prior_seed.shape[0], batch_size - prior_seed.shape[0])]),
                                    axis=0)
                                z0input = jnp.concatenate(
                                    (z0input,
                                     z0input[:min(z0input.shape[0], batch_size - z0input.shape[0])]),
                                    axis=0)

                            rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
                            sample_rng = jnp.asarray(sample_rng)
                            print((t, z0input.shape, prior_seed.shape), file=sys.stderr)
                            samples, n = sampling_fn_ode(sample_rng, pstate, z0=z0input[None, :],
                                                         prior_seed=prior_seed[None, :], init_t=jnp.asarray([t]))
                            samples = samples.reshape(
                                (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                            samples = samples[:n_dsamples ** 2]

                            sample_list = [samples[:1] * 0.]
                            sample_list.append(inverse_scaler(denoise_data))
                            for j in range(n_dsamples):
                                sample_list.append(inverse_scaler(denoise_perturbed_data[j:j + 1]))
                                sample_list.append(samples[j * n_dsamples:(j + 1) * n_dsamples])

                            samples = jnp.concatenate(sample_list, axis=0)

                            image_grid = samples.reshape((-1, *samples.shape[-3:]))
                            print(("store", os.path.join(eval_dir, f'denoise_sample_{t}.png')), file=sys.stderr)
                            with tf.io.gfile.GFile(
                                    os.path.join(eval_dir, f'denoise_sample_{t}.png'), "wb") as fout:
                                utils.save_image(image_grid, fout, nrow=n_dsamples + 1, padding=2)

                        for del_key in ['forced_t', 'forced_image', 'z0']:
                            if del_key in eval_batch:
                                del eval_batch[del_key]

                    if config.eval.enable_sampling and x0_input:
                        apply_sampling = True
                        prior_seed = None
                        z0 = jnp.concatenate([z0[i] for i in range(z0.shape[0])], axis=0)
                        z0_first_only = jnp.concatenate([z0[:, :1, :] for _ in range(z0.shape[1])], axis=1)
                        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
                        sample_rng = jnp.asarray(sample_rng)
                        first_only = False
                        if i == 0:
                            z0_vals = np.linspace(-1., 1., 10)  # [-1e-6, -3e-7, -1e-7, 0, 1e-7, 3e-7, 1e-6]
                            n_z0_vals = len(z0_vals)
                            latent_dims = list(range(2))
                            n_latent_dims = len(latent_dims)
                            sparse_z0 = False
                            if sparse_z0:
                                z0_p1 = np.zeros(z0[:, :n_z0_vals * n_latent_dims, :].shape)
                                for i1 in range(n_z0_vals):
                                    for i2 in range(n_latent_dims):
                                        z0_p1[0, i1 * n_latent_dims + i2, latent_dims[i2]] = z0_vals[i1]
                                z0_p1 = jnp.asarray(z0_p1)
                                z0 = jnp.concatenate((z0_p1, z0[:, z0_p1.shape[1]:, :]), axis=1)
                            elif n_latent_dims == 2:
                                z0_p1 = np.zeros(z0[:, :n_z0_vals * n_z0_vals, :].shape)
                                for i1 in range(n_z0_vals):
                                    for i2 in range(n_z0_vals):
                                        z0_p1[0, i1 * n_z0_vals + i2, 0] = z0_vals[i2]
                                        z0_p1[0, i1 * n_z0_vals + i2, 1] = z0_vals[n_z0_vals - i1 - 1]
                                z0_p1 = jnp.asarray(z0_p1)
                                z0 = jnp.concatenate((z0_p1, z0[:, z0_p1.shape[1]:, :]), axis=1)
                            # prior_seed = eval_batch['image'][0] * 0.
                            else:
                                z0_p1 = np.zeros(z0[:, :n_z0_vals * n_z0_vals * n_z0_vals, :].shape)
                                for i1 in range(n_z0_vals):
                                    for i2 in range(n_z0_vals):
                                        for i3 in range(n_z0_vals):
                                            row = i1 * n_z0_vals * n_z0_vals + i2 * n_z0_vals + i3
                                            z0_p1[0, row, 0] = z0_vals[i1]
                                            z0_p1[0, row, 1] = z0_vals[i2]
                                            z0_p1[0, row, 2] = z0_vals[i3]
                                z0_p1 = jnp.asarray(z0_p1)
                                z0 = jnp.concatenate((z0_p1, z0[:, z0_p1.shape[1]:, :]), axis=1)
                            # prior_seed = eval_batch['image'][0] * 0.
                        elif i == 1:
                            z0 = z0_first_only
                        elif i == 2:
                            pass
                        else:
                            apply_sampling = False
                        if apply_sampling:
                            logging.info("Sampling at %dth step" % (i + 1))
                            z0input = z0_first_only if first_only else z0
                            prior_seed = None  # eval_batch['image'][0] * 0.
                            random_prior_seed = False
                            if random_prior_seed:
                                rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
                                prior_seed = jax.random.normal(rng, shape=eval_batch['image'].shape[-3:])
                                prior_seed = jnp.concatenate(
                                    [prior_seed[None, :] for _ in range(eval_batch['image'].shape[-4])], axis=0)
                            samples, n = sampling_fn(sample_rng, pstate, z0=z0input, prior_seed=prior_seed)
                            # raise Exception((z0input.shape, prior_seed.shape))
                            samples = samples.reshape(
                                (-1, config.data.image_size, config.data.image_size, config.data.num_channels))

                            key = 'samples'
                            latent[key] = samples if key not in latent else jnp.concatenate((latent[key], samples))
                            key = 'z0input'
                            latent[key] = z0input if key not in latent else jnp.concatenate((latent[key], z0input))
                            # key = 'prior_seed'
                            # prior_seed = prior_seed if prior_seed is not None else False
                            # latent[key] = prior_seed if key not in latent else jnp.concatenate((latent[key], prior_seed))
                            if config.data.image_size == 28:
                                pred_labels_samples = predict_labels(samples)
                                pred_labels = predict_labels(eval_batch['image'][0][0])
                                key = 'pred_labels_samples'
                                latent[key] = pred_labels_samples if key not in latent else jnp.concatenate(
                                    (latent[key], pred_labels_samples))
                                key = 'pred_labels'
                                latent[key] = pred_labels if key not in latent else jnp.concatenate(
                                    (latent[key], pred_labels))

                            image_grid = samples
                            nrow = int(np.sqrt(image_grid.shape[0]))
                            with tf.io.gfile.GFile(
                                    os.path.join(eval_dir, f"ckpt_{ckpt}_samples_{i}.png"), "wb") as fout:
                                utils.save_image(image_grid, fout, nrow=nrow, padding=2)

                    if (i + 1) % 10 == 0 and jax.host_id() == 0:
                        logging.info("Finished %dth step latent evaluation" % (i + 1))

                    if store_batchwise and store_latent:
                        # Save loss values to disk or Google Cloud Storage
                        with tf.io.gfile.GFile(os.path.join(latent_dir, f"ckpt_{ckpt}_latent_{ds_str}_{i}.npz"),
                                               "wb") as fout:
                            io_buffer = io.BytesIO()
                            np.savez_compressed(io_buffer, **latent)
                            fout.write(io_buffer.getvalue())
                        latent = {}
                    # raise Exception()

                if not store_batchwise and store_latent:
                    print(("store, batchwise:", store_batchwise), file=sys.stderr)
                    print(("latent keys:", latent.keys()), file=sys.stderr)
                    with tf.io.gfile.GFile(os.path.join(latent_dir, f"ckpt_{ckpt}_latent_{ds_str}.npz"),
                                           "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, **latent)
                        fout.write(io_buffer.getvalue())

                # store latentmap + plot
                fig = plt.figure()
                plt.yscale('log')
                loss_info = jnp.asarray(
                    [[t, jnp.concatenate(latent_map['loss'][t], axis=0).mean()] for t in latent_map['loss'].keys()])
                plt.plot(loss_info[:, 0], loss_info[:, 1], label='loss')
                loss_reg_info = jnp.asarray(
                    [[t, jnp.concatenate(latent_map['reg_loss'][t], axis=0).mean()] for t in
                     latent_map['reg_loss'].keys()])
                if loss_reg_info[0][1] > 0.:
                    plt.plot(loss_reg_info[:, 0], loss_reg_info[:, 1], label='reg_loss')
                loss_reconstr_info = jnp.asarray(
                    [[t, jnp.concatenate(latent_map['reconstr_loss'][t], axis=0).mean()] for t in
                     latent_map['reconstr_loss'].keys()])
                if loss_reconstr_info[0][1] > 0.:
                    plt.plot(loss_reconstr_info[:, 0], loss_reconstr_info[:, 1], label='reconstr_loss')
                plt.legend()
                plt.savefig(os.path.join(eval_dir, f"ckpt_{ckpt}_loss_{ds_str}.png"),
                            bbox_inches='tight')
                plt.close()

                if x0_input:
                    latent_map = {
                        'z0': jnp.concatenate(latent_map['z0'], axis=0).reshape(-1, config.model.latent_input_dim),
                        'z0_mean': jnp.concatenate(latent_map['z0_mean'], axis=0).reshape(-1,
                                                                                          config.model.latent_input_dim) if len(
                            latent_map['z0_mean']) > 0 else jnp.array([]),
                        'label': jnp.concatenate(latent_map['label'], axis=0).reshape(-1),
                        'x0': jnp.concatenate(latent_map['x0'], axis=0).reshape(-1, config.data.num_channels) if len(
                            latent_map['x0']) > 0 else jnp.array([])
                    }
                    #
                    # with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_latentmap_{ds_str}.npz"),
                    #                       "wb") as fout:
                    #    io_buffer = io.BytesIO()
                    #    np.savez_compressed(io_buffer, **latent_map)
                    #    fout.write(io_buffer.getvalue())
                    #
                    # raise Exception([(key, val.shape) for (key, val) in latent_map.items()])
                    fig = plt.figure(figsize=(3, 3))
                    if config.model.latent_input_dim > 1:
                        plt.scatter(latent_map['z0'][:, 0], latent_map['z0'][:, 1], c=latent_map['label'], s=0.2)
                        plt.xlabel(r'$z_1$')
                        plt.ylabel(r'$z_2$', rotation=0)
                    elif config.data.image_size <= 1:
                        plt.scatter(latent_map['x0'][:, 0], latent_map['x0'][:, 1], c=latent_map['z0'].reshape(-1),
                                    s=0.2)
                    # plt.savefig(os.path.join(eval_dir, f"ckpt_{ckpt}_latentmap_{ds_str}.pdf"), bbox_inches='tight')
                    plt.savefig(os.path.join(eval_dir, f"ckpt_{ckpt}_latentmap_{ds_str}.png"), bbox_inches='tight')
                    plt.close()
                    if not deterministic_latent_input:
                        fig = plt.figure(figsize=(3, 3))
                        if config.model.latent_input_dim > 1:
                            plt.scatter(latent_map['z0_mean'][:, 0], latent_map['z0_mean'][:, 1], c=latent_map['label'],
                                        s=0.2)
                            plt.xlabel(r'$z_1$')
                            plt.ylabel(r'$z_2$', rotation=0)
                        else:
                            plt.scatter(latent_map['x0'][:, 0], latent_map['x0'][:, 1],
                                        c=latent_map['z0_mean'].reshape(-1),
                                        s=0.2)
                        # plt.savefig(os.path.join(eval_dir, f"ckpt_{ckpt}_latentmapmean_{ds_str}.pdf"),
                        #            bbox_inches='tight')
                        plt.savefig(os.path.join(eval_dir, f"ckpt_{ckpt}_latentmapmean_{ds_str}.png"),
                                    bbox_inches='tight')
                        plt.close()

                    info = {}
                    for key in latent.keys():
                        if key in ['samples', 'grid_score', 'grid_xx', 'grid_yy', 'z0input', 'grid_lambda_xt',
                                   'pred_labels_samples', 'pred_labels']:
                            info[key] = latent[key]
                            continue
                        key_info = {}
                        latent_info_key = latent[key]  # .item()
                        for t in latent_info_key.keys():
                            key_info[t] = latent_info_key[t]
                            key_info[t] = key_info[t].reshape((-1, *key_info[t].shape[2:]))
                            if key in ['score', 'score_unscaled', 'perturbed_data', 'mean']:
                                key_info[t] = key_info[t].reshape(key_info[t].shape[0], -1)
                        info[key] = key_info
                    ts = list(info['latent'].keys())
                    if ds_str == 'train':
                        info_train = info
                    else:
                        info_eval = info

                    # create gif of latent representation
                    if ds_str == 'eval':
                        import os
                        figure_dir = os.path.join(eval_dir, 'gif_figs')
                        if not os.path.exists(figure_dir):
                            os.mkdir(figure_dir)

                        filenames = []
                        for i, t in enumerate(reversed(ts)):
                            fig = plt.figure()
                            z = info['latent'][t]
                            plt.scatter(z[:, 0], z[:, 1], c=info['label'][t], s=0.3)
                            plt.title(f't={round(float(t), 3)}')
                            plt.xlim(-5, 5)
                            plt.ylim(-5, 5)

                            # create file name and append it to a list
                            filename = f'{figure_dir}/score_latent_{i}.png'
                            filenames.append(filename)

                            # save frame
                            plt.savefig(filename)
                            plt.close(fig)
                        plt.close()

                        import imageio
                        # build gif
                        with imageio.get_writer(os.path.join(eval_dir, f'score_latent_{ckpt}_{ds_str}.gif'), mode='I',
                                                duration=0.2) as writer:
                            for filename in filenames:
                                image = imageio.imread(filename)
                                writer.append_data(image)

                        import shutil
                        shutil.rmtree(figure_dir)

                        color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                                          '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                                          '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                                          '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
                        plt.figure(figsize=(10, 10))
                        for i in range(10, 20):
                            plt.plot([info['latent'][t][i][0] for t in ts], [info['latent'][t][i][1] for t in ts],
                                     c=color_sequence[int(info['label'][ts[0]][i])], linewidth=2.5)
                            plt.scatter([info['latent'][ts[0]][i][0]], [info['latent'][ts[0]][i][1]], c='black')
                        plt.gca().set_aspect("equal")
                        plt.savefig(os.path.join(eval_dir, f"ckpt_{ckpt}_traj_{ds_str}.png"),
                                    bbox_inches='tight')
                        plt.close()

            if getattr(config.eval, "enable_classifier", False):
                from sklearn.svm import SVC
                accs = []
                for t in ts:
                    model = SVC().fit(np.array(info_train['latent'][t]), np.array(info_train['label'][t]))
                    pred = model.predict(np.array(info_eval['latent'][t]))
                    acc = np.mean(np.array(info_eval['label'][t]) == pred)
                    logging.info((t, acc))
                    accs.append(acc)
                np.save(os.path.join(eval_dir, f"ckpt_{ckpt}_clfacc_{ds_str}"),
                        {'ts': np.array(ts), 'accs': np.array(accs)})
                plt.figure()
                plt.plot(ts, accs)
                plt.savefig(os.path.join(eval_dir, f"ckpt_{ckpt}_clfacc_{ds_str}.png"),
                            bbox_inches='tight')
                plt.close()

        # Compute the loss function on the full evaluation dataset if loss computation is enabled
        if config.eval.enable_loss:
            print(("enable_loss:",), file=sys.stderr)
            all_losses = []
            eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
            for i, batch in enumerate(eval_iter):
                eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
                eval_batch['label'] = inverse_scaler(eval_batch['label'])
                rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
                next_rng = jnp.asarray(next_rng)
                (_, _), (p_eval_loss, _) = p_eval_step((next_rng, pstate), eval_batch)
                eval_loss = flax.jax_utils.unreplicate(p_eval_loss)
                all_losses.extend(eval_loss)
                if (i + 1) % 1000 == 0 and jax.host_id() == 0:
                    logging.info("Finished %dth step loss evaluation" % (i + 1))

            # Save loss values to disk or Google Cloud Storage
            all_losses = jnp.asarray(all_losses)
            with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
                fout.write(io_buffer.getvalue())

        # Compute log-likelihoods (bits/dim) if enabled
        if config.eval.enable_bpd:
            print(("enable_bpd:",), file=sys.stderr)
            bpds = []
            begin_repeat_id = begin_bpd_round // len(ds_bpd)
            begin_batch_id = begin_bpd_round % len(ds_bpd)
            print(("begin_repeat_id:", begin_repeat_id), file=sys.stderr)
            print(("begin_batch_id:", begin_batch_id), file=sys.stderr)
            print(("len(ds_bpd):", len(ds_bpd)), file=sys.stderr)
            i_batches = 0

            # Repeat multiple times to reduce variance when needed
            for repeat in range(begin_repeat_id, bpd_num_repeats):
                bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
                for _ in range(begin_batch_id):
                    next(bpd_iter)
                print(("repeat:", repeat), file=sys.stderr)
                print(("begin_batch_id:", begin_batch_id), file=sys.stderr)
                print(("len(ds_bpd):", len(ds_bpd)), file=sys.stderr)
                for batch_id in range(begin_batch_id, len(ds_bpd)):
                    i_batches += 1
                    print(("batch_id:", batch_id), file=sys.stderr)
                    batch = next(bpd_iter)
                    eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
                    eval_batch['label'] = inverse_scaler(eval_batch['label'])
                    rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
                    step_rng = jnp.asarray(step_rng)
                    bpd, z, traj = likelihood_fn(step_rng, pstate, eval_batch['image'])
                    print(("eval_batch['image'].shape", eval_batch['image'].shape), file=sys.stderr)
                    bpd = bpd.reshape(-1)
                    bpds.extend(bpd)
                    logging.info(
                        "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (
                            ckpt, repeat, batch_id, jnp.mean(jnp.asarray(bpds))))
                    bpd_round_id = batch_id + len(ds_bpd) * repeat
                    # Save bits/dim to disk or Google Cloud Storage
                    with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                        f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                           "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, bpd)
                        fout.write(io_buffer.getvalue())
                    store_z = False
                    if store_z:
                        with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                            f"{config.eval.bpd_dataset}_ckpt_{ckpt}_z_{bpd_round_id}.npz"),
                                               "wb") as fout:
                            io_buffer = io.BytesIO()
                            np.savez_compressed(io_buffer, z)
                            fout.write(io_buffer.getvalue())
                        with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                            f"{config.eval.bpd_dataset}_ckpt_{ckpt}_label_{bpd_round_id}.npz"),
                                               "wb") as fout:
                            io_buffer = io.BytesIO()
                            np.savez_compressed(io_buffer, eval_batch['label'])
                            fout.write(io_buffer.getvalue())
                    store_trajectory = config.data.image_size <= 1
                    if store_trajectory:
                        t, y = traj
                        info = {'t': t, 'y': y, 'label': eval_batch['label']}
                        with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                            f"{config.eval.bpd_dataset}_ckpt_{ckpt}_traj_{bpd_round_id}.npz"),
                                               "wb") as fout:
                            io_buffer = io.BytesIO()
                            np.savez_compressed(io_buffer, info)
                            fout.write(io_buffer.getvalue())

                    eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=bpd_round_id, rng=rng)
                    # Save intermediate states to resume evaluation after pre-emption
                    if save_checkpoints:
                        checkpoints.save_checkpoint(
                            eval_dir,
                            eval_meta,
                            step=ckpt * (num_sampling_rounds + num_bpd_rounds) + bpd_round_id,
                            keep=1,
                            prefix=f"meta_{jax.host_id()}_")
                    if config.data.image_size > 1 and i_batches > 3:
                        break
        else:
            # Skip likelihood computation and save intermediate states for pre-emption
            eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=num_bpd_rounds - 1)
            if save_checkpoints:
                checkpoints.save_checkpoint(
                    eval_dir,
                    eval_meta,
                    step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_bpd_rounds - 1,
                    keep=1,
                    prefix=f"meta_{jax.host_id()}_")

        # Generate samples and compute IS/FID/KID when enabled
        if config.eval.enable_sampling and not x0_input:
            print(("sampling:",), file=sys.stderr)
            state = jax.device_put(state)
            # Run sample generation for multiple rounds to create enough samples
            # Designed to be pre-emption safe. Automatically resumes when interrupted
            for r in range(begin_sampling_round, num_sampling_rounds):
                if jax.host_id() == 0:
                    logging.info("sampling -- ckpt: %d, round: %d of %d" % (ckpt, r, num_sampling_rounds))

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(
                    eval_dir, f"ckpt_{ckpt}_host_{jax.host_id()}")
                tf.io.gfile.makedirs(this_sample_dir)
                rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
                sample_rng = jnp.asarray(sample_rng)

                if not os.path.exists(os.path.join(this_sample_dir, f"samples_{r}.npz")):
                    if use_vae_samples:
                        prior_seed = jnp.asarray(np.load(f'../nvae/samples_np_50/samples_{r}.npy'))
                        prior_seed = prior_seed.reshape((1, *prior_seed.shape))
                        prior_seed = scaler(prior_seed)
                        init_t = vae_sample_t
                        print(("prior_seed.shape:", prior_seed.shape), file=sys.stderr)
                        if vae_sample_is_p0:
                            rng, step_rng = jax.random.split(rng)
                            z = jax.random.normal(step_rng, prior_seed.shape)
                            mean, std = sde.marginal_prob(prior_seed, init_t, params=state.params_ema)
                            prior_seed = mean + std * z
                        print(("prior_seed_after.shape:", prior_seed.shape), file=sys.stderr)
                        samples, n = sampling_fn(sample_rng, pstate, init_t=jnp.asarray([init_t]),
                                                 prior_seed=prior_seed)
                    else:
                        # init_t = 1.
                        # samples, n = sampling_fn(sample_rng, pstate, init_t=jnp.asarray([init_t]))
                        if conditional_sampling:
                            rng, step_rng = jax.random.split(rng)
                            labels = jax.random.choice(step_rng, jnp.array(list(range(10))),
                                                       (jax.local_device_count(), config.eval.batch_size,))
                            # labels = jnp.ones((jax.local_device_count(), config.eval.batch_size), dtype=jnp.int32) * labels
                            samples, n = sampling_fn(sample_rng, pstate, labels)
                        else:
                            samples, n = sampling_fn(sample_rng, pstate)
                    if not hasattr(config.sampling, 'clip_to_255') or config.sampling.clip_to_255:
                        samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
                    else:
                        raise Exception()
                    samples = samples.reshape(
                        (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                    # Write samples to disk or Google Cloud Storage
                    with tf.io.gfile.GFile(
                            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, samples=samples)
                        fout.write(io_buffer.getvalue())

                    # Force garbage collection before calling TensorFlow code for Inception network
                    gc.collect()
                    if eval_samples:
                        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                                       inceptionv3=inceptionv3)
                        # Force garbage collection again before returning to JAX code
                        gc.collect()
                        # Save latent represents of the Inception network to disk or Google Cloud Storage
                        with tf.io.gfile.GFile(
                                os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
                            io_buffer = io.BytesIO()
                            np.savez_compressed(
                                io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
                            fout.write(io_buffer.getvalue())

                        # Update the intermediate evaluation state
                        eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=r, rng=rng)
                        # Save an intermediate checkpoint directly if not the last round.
                        # Otherwise save eval_meta after computing the Inception scores and FIDs
                        if r < num_sampling_rounds - 1 and save_checkpoints:
                            checkpoints.save_checkpoint(
                                eval_dir,
                                eval_meta,
                                step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
                                keep=1,
                                prefix=f"meta_{jax.host_id()}_")

                if eval_samples and (r % 10 == 0 and r > 0):
                    # Compute inception scores, FIDs and KIDs.
                    if jax.host_id() == 0:
                        # Load all statistics that have been previously computed and saved for each host
                        all_logits = []
                        all_pools = []
                        for host in range(jax.host_count()):
                            this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{host}")

                            for r2 in range(r):
                                stats = tf.io.gfile.glob(os.path.join(this_sample_dir, f'statistics_{r2}.npz'))
                                for stat_file in stats:
                                    with tf.io.gfile.GFile(stat_file, "rb") as fin:
                                        stat = np.load(fin)
                                        if not inceptionv3:
                                            all_logits.append(stat["logits"])
                                        all_pools.append(stat["pool_3"])

                        if not inceptionv3:
                            all_logits = np.concatenate(
                                all_logits, axis=0)[:config.eval.num_samples]
                        all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

                        # Load pre-computed dataset statistics.
                        data_stats = evaluation.load_dataset_stats(config)
                        data_pools = data_stats["pool_3"]

                        # Compute FID/KID/IS on all samples together.
                        if not inceptionv3:
                            inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
                        else:
                            inception_score = -1

                        fid = tfgan.eval.frechet_classifier_distance_from_activations(
                            data_pools, all_pools)
                        # Hack to get tfgan KID work for eager execution.
                        tf_data_pools = tf.convert_to_tensor(data_pools)
                        tf_all_pools = tf.convert_to_tensor(all_pools)
                        kid = tfgan.eval.kernel_classifier_distance_from_activations(
                            tf_data_pools, tf_all_pools).numpy()
                        del tf_data_pools, tf_all_pools

                        logging.info(
                            f"ckpt-%d --- %d samples, FID: {fid.numpy().round(2)}, inception_score: %.6e, FID: %.6e, KID: %.6e" % (
                                ckpt, r, inception_score, fid, kid))

                        with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}_{r}.npz"),
                                               "wb") as f:
                            io_buffer = io.BytesIO()
                            np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
                            f.write(io_buffer.getvalue())
                    else:
                        # For host_id() != 0.
                        # Use file existence to emulate synchronization across hosts
                        while not tf.io.gfile.exists(os.path.join(eval_dir, f"report_{ckpt}.npz")):
                            time.sleep(1.)

            if eval_samples:
                # Compute inception scores, FIDs and KIDs.
                if jax.host_id() == 0:
                    # Load all statistics that have been previously computed and saved for each host
                    all_logits = []
                    all_pools = []
                    for host in range(jax.host_count()):
                        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{host}")

                        stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
                        wait_message = False
                        while len(stats) < num_sampling_rounds:
                            if not wait_message:
                                logging.warning("Waiting for statistics on host %d" % (host,))
                                logging.warning("Waiting because {} < {}".format(len(stats), num_sampling_rounds))
                                wait_message = True
                            stats = tf.io.gfile.glob(
                                os.path.join(this_sample_dir, "statistics_*.npz"))
                            time.sleep(30)

                        for stat_file in stats:
                            with tf.io.gfile.GFile(stat_file, "rb") as fin:
                                stat = np.load(fin)
                                if not inceptionv3:
                                    all_logits.append(stat["logits"])
                                all_pools.append(stat["pool_3"])

                    if not inceptionv3:
                        all_logits = np.concatenate(
                            all_logits, axis=0)[:config.eval.num_samples]
                    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

                    # Load pre-computed dataset statistics.
                    data_stats = evaluation.load_dataset_stats(config)
                    data_pools = data_stats["pool_3"]

                    # Compute FID/KID/IS on all samples together.
                    if not inceptionv3:
                        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
                    else:
                        inception_score = -1

                    fid = tfgan.eval.frechet_classifier_distance_from_activations(
                        data_pools, all_pools)
                    # Hack to get tfgan KID work for eager execution.
                    tf_data_pools = tf.convert_to_tensor(data_pools)
                    tf_all_pools = tf.convert_to_tensor(all_pools)
                    kid = tfgan.eval.kernel_classifier_distance_from_activations(
                        tf_data_pools, tf_all_pools).numpy()
                    del tf_data_pools, tf_all_pools

                    logging.info(
                        "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
                            ckpt, inception_score, fid, kid))

                    with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                                           "wb") as f:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
                        f.write(io_buffer.getvalue())
                else:
                    # For host_id() != 0.
                    # Use file existence to emulate synchronization across hosts
                    while not tf.io.gfile.exists(os.path.join(eval_dir, f"report_{ckpt}.npz")):
                        time.sleep(1.)

            # Save eval_meta after computing IS/KID/FID to mark the end of evaluation for this checkpoint
            if save_checkpoints:
                checkpoints.save_checkpoint(
                    eval_dir,
                    eval_meta,
                    step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
                    keep=1,
                    prefix=f"meta_{jax.host_id()}_")

        else:
            # Skip sampling and save intermediate evaluation states for pre-emption
            eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=num_sampling_rounds - 1, rng=rng)
            if save_checkpoints:
                checkpoints.save_checkpoint(
                    eval_dir,
                    eval_meta,
                    step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_sampling_rounds - 1 + num_bpd_rounds,
                    keep=1,
                    prefix=f"meta_{jax.host_id()}_")

        begin_bpd_round = 0
        begin_sampling_round = 0

    # Remove all meta files after finishing evaluation
    meta_files = tf.io.gfile.glob(
        os.path.join(eval_dir, f"meta_{jax.host_id()}_*"))
    for file in meta_files:
        tf.io.gfile.remove(file)
