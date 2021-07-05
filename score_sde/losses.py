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

"""All functions related to loss computation and optimization.
"""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from utils import batch_mul
import sys


def get_optimizer(config):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = flax.optim.Adam(beta1=config.optim.beta1, eps=config.optim.eps,
                                    weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(state,
                    grad,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        lr = state.lr
        if warmup > 0:
            lr = lr * jnp.minimum(state.step / warmup, 1.0)
        if grad_clip >= 0:
            # Compute global gradient norm
            grad_norm = jnp.sqrt(
                sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
            # Clip gradient
            clipped_grad = jax.tree_map(
                lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
        else:  # disabling gradient clipping if grad_clip < 0
            clipped_grad = grad
        return state.optimizer.apply_gradient(clipped_grad, learning_rate=lr)

    return optimize_fn


def lambda_optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(state,
                    grad,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        lr = state.lambda_lr
        if warmup > 0:
            lr = lr * jnp.minimum(state.lambda_step / warmup, 1.0)
        if grad_clip >= 0:
            # Compute global gradient norm
            grad_norm = jnp.sqrt(
                sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
            # Clip gradient
            clipped_grad = jax.tree_map(
                lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
        else:  # disabling gradient clipping if grad_clip < 0
            clipped_grad = grad
        return state.lambda_optimizer.apply_gradient(clipped_grad, learning_rate=lr)

    return optimize_fn


def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5,
                    lambda_z=None, adversarial=False, lambda_model=None, x0_input=False,
                    deterministic_latent_input=True, latent_dim=-1, single_t=None, config=None):
    print(("get_sde_loss_fn"), file=sys.stderr)
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch, lambda_params=None, lambda_states=None, step=None,
                lambda_reconstr_balanced=None):
        """Compute the loss function.

        Args:
          rng: A JAX random state.
          params: A dictionary that contains trainable parameters of the score-based model.
          states: A dictionary that contains mutable states of the score-based model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
          new_model_state: A dictionary that contains the mutated states of the score-based model.
        """

        score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous,
                                       return_state=True, config=config)
        data = batch['image']

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=1)

        random_normal = None
        if not deterministic_latent_input and latent_dim > 0:
            rng, step_rng = random.split(rng)
            random_normal = random.normal(step_rng, (data.shape[0], latent_dim))

        mle_sampling = False
        if mle_sampling:
            a = params['sde_beta_min']
            b = params['sde_beta_max']
            mle_sampling *= jnp.absolute(a - b) >= 1e-2
            # only apply mle_sampling if a and b are not close
            t = t * (1 - mle_sampling) + mle_sampling * (a - jnp.sqrt((b * b - a * a) * t + a * a)) / jnp.where(
                jnp.absolute(a - b) >= 1e-2, a - b, 1)

        lambda_method_sampling = getattr(config.training, 'lambda_method_sampling', False)

        lambda_model_sampling = lambda_model is not None and not adversarial and lambda_method_sampling
        lm_type = getattr(config.training, 'lm_type', '')
        if lambda_model_sampling:
            t_uni = t
            if lm_type == 'mog':
                t_new = jnp.zeros_like(t)
                n_gaussians = config.training.lm_n_mog
                z = random.normal(step_rng, t_new.shape)
                for i in range(n_gaussians):
                    idx = (t > (i + 0.) / n_gaussians) & (t <= (i + 1.) / n_gaussians)
                    t_new += idx * (lambda_params[f'lambda_mean_{i}'] + z * lambda_params[f'lambda_std_{i}'])

                # replace out of bounds values with uniform t
                in_bounds = (t_new >= eps) & (t_new <= 1.)
                t = t_new * in_bounds + t * (1 - in_bounds)
            elif lm_type in ['linear', 'powera', 'power2', 'power4', 'power16']:
                # set t back to uniform between 0 and 1
                t = (t - eps) / (1 - eps)
                if lm_type == 'linear':
                    a = nn.sigmoid(lambda_params['a']) * 2 - 1
                    apply_sampling = jnp.absolute(a) >= 1e-2
                    t_lin = (a - 1 + jnp.sqrt((1 - a) ** 2 + 4 * a * t)) / jnp.where(jnp.absolute(a) > 1e-2, 2 * a, 1)
                    t = t_lin * apply_sampling + t * (1 - apply_sampling)
                elif lm_type in ['powera', 'power2', 'power4', 'power16']:
                    if lm_type == 'powera':
                        a = jnp.exp(lambda_params['a'])
                    elif lm_type == 'power2':
                        a = nn.sigmoid(lambda_params['a']) * 2
                    elif lm_type == 'power4':
                        a = nn.sigmoid(lambda_params['a']) * 4
                    elif lm_type == 'power16':
                        a = nn.sigmoid(lambda_params['a']) * 16
                    else:
                        raise Exception('error')
                    t = 1 - (1 - t) ** (1. / (a + 1))
                # set t back to range eps to 1
                t = t * (1 - eps) + eps
            else:
                raise Exception('no lm type?')
            # add certain percentage of uniform t
            rng, step_rng = random.split(rng)
            is_uni = random.bernoulli(step_rng, config.training.lm_basep, t.shape)
            t = t_uni * is_uni + t * (1 - is_uni)

        lambda_method = getattr(config.training, 'lambda_method', '')
        if lambda_method == 'squared':
            t = t ** 2
        elif lambda_method == 'antisquared':
            t = 1 - ((1 - t) ** 2)
        elif lambda_method == 'squaredepsadjusted':
            t = (t - eps) / (1 - eps)
            t = t ** 2
            t = t * (1 - eps) + eps
        elif lambda_method == 'lossprop':
            if lambda_method_sampling:
                rng, step_rng = random.split(rng)
                n_ts = lambda_reconstr_balanced.shape[-1]
                t_i = random.choice(step_rng, jnp.asarray(list(range(n_ts))), shape=t.shape,
                                    p=lambda_reconstr_balanced / lambda_reconstr_balanced.sum())
                rng, step_rng = random.split(rng)
                t = t_i / n_ts + random.uniform(step_rng, t.shape, minval=0, maxval=1. / n_ts)

        p_eps = getattr(config.training, 'p_eps', 0.0)
        if p_eps > 0.:
            rng, step_rng = random.split(rng)
            is_eps = random.bernoulli(step_rng, p_eps, t.shape)
            t = t * (1 - is_eps) + eps * is_eps

        if hasattr(config.model, 'uniform_std_sampling') and config.model.uniform_std_sampling:
            assert (config.training.sde == 'vesde')
            sigma_min = config.model.sigma_min
            sigma_max = config.model.sigma_max
            t_std = random.uniform(step_rng, (data.shape[0],), minval=sigma_min, maxval=sigma_max)
            t = jnp.log(t_std / sigma_min) / jnp.log(sigma_max / sigma_min)

        if single_t is not None:
            t = t * 0. + single_t

        t = jnp.clip(t, a_min=eps, a_max=1.)
        t *= sde.T
        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, data.shape)
        # params_sde = {'beta_min': params['sde_beta_min'], 'beta_max': params['sde_beta_max']}
        mean, std = sde.marginal_prob(data, t, params=params)
        mle_coef = sde.lambda_mle(t, params=params)
        perturbed_data = mean + batch_mul(std, z)

        loss_weight = 1.

        if hasattr(config.model, 'same_xt_in_batch') and config.model.same_xt_in_batch:
            perturbed_data = jnp.concatenate([perturbed_data[:1] - mean[:1] for _ in range(perturbed_data.shape[0])],
                                             axis=0)
            z = batch_mul(perturbed_data - mean, 1. / std)

            # reweighting for importance sampling
            sq_dist = jnp.square(perturbed_data - mean).reshape(data.shape[0], -1).sum(axis=-1)
            sq_dist_to_zero = jnp.square(perturbed_data).reshape(data.shape[0], -1).sum(axis=-1)
            loss_weight = jnp.exp(-0.5 * jnp.square(1. / std) * (sq_dist - sq_dist_to_zero))
            # raise Exception((data.shape, perturbed_data.shape, z.shape))

        rng, step_rng = random.split(rng)
        model_result, new_model_state = score_fn(perturbed_data, t, rng=step_rng, x0=data if x0_input else None,
                                                 random_normal=random_normal)
        score = model_result['score']

        train_clf = getattr(config.training, 'train_clf', False)
        if train_clf:
            if train:
                # cross entropy loss
                logits = model_result['clf_logits']
                one_hot_labels = jax.nn.one_hot(batch['label'], num_classes=10)
                losses = -jnp.sum(one_hot_labels * logits, axis=-1)
            else:
                # accuracy
                logits = model_result['clf_logits']
                losses = jnp.argmax(logits, -1) == batch['label']
        else:
            if lambda_method == 'lossprop' and not lambda_method_sampling:
                loss_weight = lambda_reconstr_balanced[((t / sde.T) * lambda_reconstr_balanced.shape[-1]).astype(int)]

            if lambda_model is not None:
                lambda_model_fn = mutils.get_model_fn(lambda_model, lambda_params, lambda_states, train=train)
                labels = t * (sde.N - 1)
                if getattr(config.model, 'lambda_model_x0', False):
                    lambda_model_result, lambda_state = lambda_model_fn(data, labels, rng=step_rng)
                else:
                    lambda_model_result, lambda_state = lambda_model_fn(perturbed_data, labels, rng=step_rng)
                if adversarial:
                    new_model_state = lambda_state

                # losses = jnp.square(batch_mul(score, std) + z)
                # losses = losses.reshape((losses.shape[0], -1))
                # losses = reduce_op(losses, axis=-1)
                lambda_xt = lambda_model_result['lambda']
                if not lambda_model_sampling:
                    loss_weight = jnp.clip(lambda_xt, a_min=getattr(config.training, 'lm_basep', 0.0))

            if not likelihood_weighting:
                use_mle = False
                use_pow4 = False
                use_pow2 = False
                if getattr(config.model, 'predictx0', False):
                    losses = jnp.square(model_result['output'] - data)
                    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
                elif use_mle:
                    losses = jnp.square(score + batch_mul(z, 1. / std))
                    losses = losses.reshape((losses.shape[0], -1))
                    losses = reduce_op(losses, axis=-1)
                    losses = losses * (mle_sampling * 1 + (1 - mle_sampling) * jnp.clip(mle_coef, a_min=0))
                elif use_pow4:
                    losses = jnp.square(jnp.square(score + batch_mul(z, 1. / std)))
                    losses = losses.reshape((losses.shape[0], -1))
                    losses = reduce_op(losses, axis=-1)
                elif use_pow2:
                    losses = jnp.square(score + batch_mul(z, 1. / std))
                    losses = losses.reshape((losses.shape[0], -1))
                    losses = reduce_op(losses, axis=-1)
                else:
                    losses = jnp.square(batch_mul(score, std) + z)
                    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
                # losses = jnp.square(score + batch_mul(z, 1. / std)) -> leads to 1e7 loss, also 1e6 samples
                # losses = batch_mul(jnp.square(model_result['output'] - mean), std ** 2)
            else:
                raise NotImplementedError('not implemented')
                g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
                losses = jnp.square(score + batch_mul(z, 1. / std))
                losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

            losses *= loss_weight

            lambda_reconstr = getattr(config.model, 'lambda_reconstr', 0.0)
            lambda_reconstr_rate = getattr(config.model, 'lambda_reconstr_rate', 0.0)
            if lambda_reconstr != 0. or lambda_reconstr_rate != 0.:
                # losses_rec = jnp.square(model_result['reconstruction'] - data)
                data_reconstr = data
                if config.data.centered:
                    data_reconstr = (data_reconstr + 1) * 0.5
                #if True:
                losses_rec = jnp.square(model_result['reconstruction'] - nn.sigmoid(data_reconstr))
                #else:
                #    logits = nn.log_sigmoid(model_result['reconstruction'])
                #    losses_rec = -(data_reconstr * logits + (1. - data_reconstr) * jnp.log(-jnp.expm1(logits)))
                losses_rec = reduce_op(losses_rec.reshape((losses_rec.shape[0], -1)), axis=-1)
                if lambda_reconstr > 0.:
                    losses += lambda_reconstr * losses_rec
                elif lambda_reconstr_rate > 0.:
                    losses += lambda_reconstr_balanced[
                                  ((t / sde.T) * lambda_reconstr_balanced.shape[-1]).astype(int)] * losses_rec
                elif lambda_reconstr < 0. or lambda_reconstr_rate < 0.:
                    # only reconstruction loss
                    losses = losses * 0. + losses_rec

            if lambda_z is not None and deterministic_latent_input:
                reg_losses = jnp.sum(jnp.absolute(model_result['latent']), axis=-1)
                if len(losses.shape) != len(reg_losses.shape) or True in [losses.shape[i] != reg_losses.shape[i] for i
                                                                          in
                                                                          range(len(losses.shape))]:
                    raise Exception((losses.shape, reg_losses.shape))
                losses += lambda_z * reg_losses
                # reg_loss = lambda_z * jnp.mean(reg_losses)
                # loss += reg_loss

            if not deterministic_latent_input and x0_input:
                assert lambda_z is not None
                logvar = model_result['z_logvar']
                kl_div = -0.5 * jnp.sum(1 + logvar - jnp.square(model_result['z_mean']) - jnp.exp(logvar))
                losses += lambda_z * kl_div

        # losses += -mle_coef * 10.
        loss = jnp.mean(losses)

        if adversarial:
            assert (not train_clf)
            loss *= -1.

            # Regularize lambda model
            means = lambda_model_result['means']
            if means is not None:
                n_gaussians = len(means)
                reg_loss_means = 0.
                for i1 in range(n_gaussians):
                    for i2 in range(i1 + 1, n_gaussians):
                        reg_loss_means += 1. / jnp.square(means[i1] - means[i2]).mean() / (
                                n_gaussians * (n_gaussians - 1) * 0.5)
                reg_loss_stds = 0.
                stds = lambda_model_result['stds']
                for std in stds:
                    reg_loss_stds += (1. / std).mean() / n_gaussians

                temperature_weight = 1.
                include_temperature = False
                if include_temperature:
                    temperature_weight = 10. ** (1. * jnp.clip((100000. - step) / 100000, a_min=0.))

                loss += reg_loss_means * getattr(config.training, 'lm_mean_reg', 3e-5) * temperature_weight
                loss += reg_loss_stds * getattr(config.training, 'lm_std_reg', 3e-4) * temperature_weight

        return loss, new_model_state
        # return {'loss': loss, 'new_model_state': new_model_state, 'latent': model_result['latent']}

    return loss_fn


def get_sde_latent_fn(sde, model, eps=1e-5, lambda_model=None, x0_input=False, deterministic_latent_input=False,
                      latent_dim=-1, config=None):
    def latent_fn(rng, params, states, batch, lambda_params=None, lambda_states=None, z0=None,
                  lambda_reconstr_balanced=None):
        score_fn = mutils.get_score_fn(sde, model, params, states, train=False, continuous=True, return_state=True,
                                       config=config)
        data = batch['image']

        reduce_op = jnp.mean

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
        forced_t = False
        if 'forced_t' in batch:
            forced_t = batch['forced_t'].reshape(t.shape)
            t = t * 0. + (batch['forced_t'].reshape(t.shape) * sde.T).clip(a_min=eps, a_max=sde.T)
        mle_sampling = False
        t_orig, a, b = False, False, False
        if mle_sampling:
            a = params['sde_beta_min']
            b = params['sde_beta_max']
            mle_sampling *= jnp.absolute(a - b) >= 1e-2
            t_orig = t
            # only apply mle_sampling if a and b are not close
            t = t * (1 - mle_sampling) + mle_sampling * (a - jnp.sqrt((b * b - a * a) * t + a * a)) / jnp.where(
                jnp.absolute(a - b) >= 1e-2, a - b, 1)

        random_normal = None
        if not deterministic_latent_input and latent_dim > 0:
            rng, step_rng = random.split(rng)
            random_normal = random.normal(step_rng, (data.shape[0], latent_dim))

        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, data.shape)
        mean, std = sde.marginal_prob(data, t, params=params)
        perturbed_data = mean + batch_mul(std, z)
        rng, step_rng = random.split(rng)
        model_result, new_model_state = score_fn(perturbed_data, t, rng=step_rng, x0=data if x0_input else None, z0=z0,
                                                 use_mean=False, random_normal=random_normal)
        # raise Exception((model_result['latent'].shape, model_result['output'].shape, ))
        mle_coef = sde.lambda_mle(t, params=params)
        empirical_coef = 1. / jnp.square(std)

        lambda_xt = False
        if lambda_model is not None:
            lambda_model_fn = mutils.get_model_fn(lambda_model, lambda_params, lambda_states, train=False)
            labels = t * (sde.N - 1)
            lambda_model_result, lambda_state = lambda_model_fn(perturbed_data, labels, rng=step_rng)
            lambda_xt = lambda_model_result['lambda']
            # lambda_z = lambda_model_result['latent']

        score_for_loss = model_result['score']
        if z0 is not None:
            raise Exception('where does this happen?')
            model_result_loss, _ = score_fn(perturbed_data, t, rng=step_rng, x0=data if x0_input else None,
                                            z0=None,
                                            use_mean=False, random_normal=random_normal)
            score_for_loss = model_result_loss['score']
        losses = jnp.square(batch_mul(score_for_loss, std) + z)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        dsm_losses = losses

        reconstr_losses = False
        lambda_reconstr = getattr(config.model, 'lambda_reconstr', 0.0)
        lambda_reconstr_rate = getattr(config.model, 'lambda_reconstr_rate', 0.0)
        if lambda_reconstr > 0. or lambda_reconstr_rate > 0.:
            losses_rec = jnp.square(model_result['reconstruction'] - data)
            losses_rec = reduce_op(losses_rec.reshape((losses_rec.shape[0], -1)), axis=-1)
            if lambda_reconstr > 0.:
                reconstr_losses = lambda_reconstr * losses_rec
            elif lambda_reconstr_rate > 0.:
                reconstr_losses = lambda_reconstr_balanced[((t / sde.T) * 20).astype(int)] * losses_rec
            losses += reconstr_losses

        lambda_z = getattr(config.model, 'lambda_z', None)
        reg_losses = False
        if lambda_z is not None and deterministic_latent_input:
            reg_losses = jnp.sum(jnp.absolute(model_result['latent']), axis=-1)
            if len(losses.shape) != len(reg_losses.shape) or True in [losses.shape[i] != reg_losses.shape[i] for i in
                                                                      range(len(losses.shape))]:
                raise Exception((losses.shape, reg_losses.shape))
            reg_losses *= lambda_z
            losses += reg_losses
            # reg_loss = lambda_z * jnp.mean(reg_losses)
            # loss += reg_loss

        if not deterministic_latent_input and x0_input:
            assert lambda_z is not None
            logvar = model_result['z_logvar']
            kl_div = -0.5 * jnp.sum(1 + logvar - jnp.square(model_result['z_mean']) - jnp.exp(logvar))
            reg_losses = lambda_z * kl_div
            losses += reg_losses

        return {'latent': model_result['latent'] if 'latent' in model_result else False,
                'label': batch['label'],
                't': t,
                'forced_t': forced_t,
                # 'a': a,
                # 'b': b,
                # 'mle_sampling': mle_sampling,
                # 'beta_min': params['sde_beta_min'] if 'sde_beta_min' in params else False,
                # 'score': model_result['score'],
                # 'data': data,
                # 'perturbed_data': perturbed_data,
                # 'score_unscaled': model_result['output'],
                # 'mean': mean,
                'std': std,
                'z_mean': model_result['z_mean'] if 'z_mean' in model_result else False,
                'z_logvar': model_result['z_logvar'] if 'z_logvar' in model_result else False,
                # 'perturbed_data': perturbed_data,
                'losses': losses,
                'dsm_losses': dsm_losses,
                'reg_losses': reg_losses,
                'reconstr_losses': reconstr_losses,
                'lambda_xt': lambda_xt,
                # 'mle_coef': mle_coef,
                # 'empirical_coef': empirical_coef,
                # 'lambda_z': lambda_z,
                }
        # return model_result['latent'], batch['label']

    return latent_fn


def get_sde_latentscore_fn(sde, model, eps=1e-5, lambda_model=None, x0_input=False, deterministic_latent_input=False,
                           latent_dim=-1, config=None):
    def latentscore_fn(rng, params, states, batch, lambda_params=None, lambda_states=None):
        score_fn = mutils.get_score_fn(sde, model, params, states, train=False, continuous=True, return_state=True,
                                       config=config)
        data = batch['forced_image']

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
        if 'forced_t' in batch:
            t = t * 0. + (batch['forced_t'] * sde.T).clip(a_min=eps, a_max=sde.T)

        score = False
        if not x0_input:
            rng, step_rng = random.split(rng)
            model_result, new_model_state = score_fn(data, t, rng=step_rng, x0=None,
                                                     z0=batch['z0'] if 'z0' in batch else None)

            score = model_result['score']

        lambda_xt = False
        if lambda_model is not None:
            lambda_model_fn = mutils.get_model_fn(lambda_model, lambda_params, lambda_states, train=False)
            labels = t * (sde.N - 1)
            lambda_model_result, lambda_state = lambda_model_fn(data, labels, rng=step_rng)
            lambda_xt = lambda_model_result['lambda']

        return {'score': score, 'lambda_xt': lambda_xt}

    return latentscore_fn


def get_smld_loss_fn(vesde, model, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch['image']
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
        sigmas = smld_sigma_array[labels]
        rng, step_rng = random.split(rng)
        noise = batch_mul(random.normal(step_rng, data.shape), sigmas)
        perturbed_data = noise + data
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        target = -batch_mul(noise, 1. / (sigmas ** 2))
        losses = jnp.square(score - target)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * sigmas ** 2
        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_ddpm_loss_fn(vpsde, model, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch['image']
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vpsde.N, shape=(data.shape[0],))
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, data.shape)
        perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + \
                         batch_mul(sqrt_1m_alphas_cumprod[labels], noise)
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        losses = jnp.square(score - noise)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_step_fn(sde, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False,
                lambda_z=None, adversarial=False, lambda_model=None, x0_input=False, deterministic_latent_input=True,
                latent_dim=-1, single_t=None, config=None):
    print(("get_step_fn, lambda_z=", lambda_z), file=sys.stderr)
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
      lambda_z: regularization weight of the L1-Norm of the latent space
    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        print(("should call get_sde_loss_fn", lambda_z), file=sys.stderr)

        loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                                  continuous=True, likelihood_weighting=likelihood_weighting, lambda_z=lambda_z,
                                  adversarial=adversarial, lambda_model=lambda_model, x0_input=x0_input,
                                  deterministic_latent_input=deterministic_latent_input, latent_dim=latent_dim,
                                  single_t=single_t, config=config)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(carry_state, batch):
        print(("step_fn"), file=sys.stderr)
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1 if not adversarial else 4, has_aux=True)
        grad = None
        lambda_reconstr_balanced = state.lambda_reconstr_balanced
        if train:
            params = state.optimizer.target
            states = state.model_state
            if lambda_model is None:
                (loss, new_model_state), grad = grad_fn(step_rng, params, states, batch, step=state.step,
                                                        lambda_reconstr_balanced=lambda_reconstr_balanced)
            else:
                lambda_params = state.lambda_optimizer.target
                lambda_states = state.lambda_model_state
                (loss, new_model_state), grad = grad_fn(step_rng, params, states, batch, lambda_params, lambda_states,
                                                        step=state.step,
                                                        lambda_reconstr_balanced=lambda_reconstr_balanced)
            grad = jax.lax.pmean(grad, axis_name='batch')
            new_optimizer = optimize_fn(state, grad)
            if not adversarial:
                new_params_ema = jax.tree_multimap(
                    lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
                    state.params_ema, new_optimizer.target
                )
                step = state.step + 1
                new_state = state.replace(
                    step=step,
                    optimizer=new_optimizer,
                    model_state=new_model_state,
                    params_ema=new_params_ema
                )
            else:
                new_params_ema = jax.tree_multimap(
                    lambda p_ema, p: p_ema * state.lambda_ema_rate + p * (1. - state.lambda_ema_rate),
                    state.lambda_params_ema, new_optimizer.target
                )
                step = state.lambda_step + 1
                new_state = state.replace(
                    lambda_step=step,
                    lambda_optimizer=new_optimizer,
                    lambda_model_state=new_model_state,
                    lambda_params_ema=new_params_ema
                )
        else:
            if lambda_model is None:
                loss, _ = loss_fn(step_rng, state.params_ema, state.model_state, batch, step=state.step,
                                  lambda_reconstr_balanced=lambda_reconstr_balanced)
            else:
                lambda_params = state.lambda_optimizer.target
                lambda_states = state.lambda_model_state
                loss, _ = loss_fn(step_rng, state.params_ema, state.model_state, batch, lambda_params, lambda_states,
                                  step=state.step, lambda_reconstr_balanced=lambda_reconstr_balanced)
            new_state = state

        loss = jax.lax.pmean(loss, axis_name='batch')
        new_carry_state = (rng, new_state)
        # return new_carry_state, (loss, grad)
        return new_carry_state, (loss, None)

    return step_fn


def get_step_fn_latent(sde, model, lambda_model=None, x0_input=False, latentscorefn=False,
                       deterministic_latent_input=False,
                       latent_dim=-1, config=None):
    if latentscorefn:
        latent_fn = get_sde_latentscore_fn(sde, model, lambda_model=lambda_model, x0_input=x0_input,
                                           deterministic_latent_input=deterministic_latent_input, latent_dim=latent_dim,
                                           config=config)
    else:
        latent_fn = get_sde_latent_fn(sde, model, lambda_model=lambda_model, x0_input=x0_input,
                                      deterministic_latent_input=deterministic_latent_input, latent_dim=latent_dim,
                                      config=config)

    def step_fn(carry_state, batch):
        (rng, state) = carry_state
        rng, step_rng = jax.random.split(rng)

        lambda_reconstr_balanced = state.lambda_reconstr_balanced
        if lambda_model is None:
            latent = latent_fn(step_rng, state.params_ema, state.model_state, batch,
                               lambda_reconstr_balanced=lambda_reconstr_balanced)
        else:
            lambda_params = state.lambda_optimizer.target
            lambda_states = state.lambda_model_state
            latent = latent_fn(step_rng, state.params_ema, state.model_state, batch, lambda_params, lambda_states,
                               lambda_reconstr_balanced=lambda_reconstr_balanced)

        new_state = state

        # latent = jax.lax.pmean(latent, axis_name='batch')
        new_carry_state = (rng, new_state)
        return new_carry_state, latent

    return step_fn
