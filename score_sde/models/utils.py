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

"""All functions and modules related to model definition.
"""
from typing import Any

import flax
import functools
import jax.numpy as jnp
import jax.random as random
import sde_lib
import jax
import numpy as np
from models import wideresnet_noise_conditional
from flax.training import checkpoints
from utils import batch_mul

import sys


# The dataclass that stores all training states
@flax.struct.dataclass
class State:
    step: int
    optimizer: flax.optim.Optimizer
    lr: float
    model_state: Any
    ema_rate: float
    params_ema: Any
    lambda_step: int
    lambda_optimizer: flax.optim.Optimizer
    lambda_lr: float
    lambda_model_state: Any
    lambda_ema_rate: float
    lambda_params_ema: Any
    lambda_reconstr_balanced: Any
    rng: Any


# The dataclass that stores all training states
@flax.struct.dataclass
class State2:
    step: int
    optimizer: flax.optim.Optimizer
    lr: float
    model_state: Any
    ema_rate: float
    params_ema: Any
    lambda_step: int
    lambda_optimizer: flax.optim.Optimizer
    lambda_lr: float
    lambda_model_state: Any
    lambda_ema_rate: float
    lambda_params_ema: Any
    rng: Any


# The dataclass that stores all training states
@flax.struct.dataclass
class Stateold:
    step: int
    optimizer: flax.optim.Optimizer
    lr: float
    model_state: Any
    ema_rate: float
    params_ema: Any
    rng: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = jnp.exp(
        jnp.linspace(
            jnp.log(config.model.sigma_max), jnp.log(config.model.sigma_min),
            config.model.num_scales))

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
        'beta_min': beta_start * (num_diffusion_timesteps - 1),
        'beta_max': beta_end * (num_diffusion_timesteps - 1),
        'num_diffusion_timesteps': num_diffusion_timesteps
    }


def init_model(rng, config, x0_input=False):
    """ Initialize a `flax.linen.Module` model. """
    model_name = config.model.name
    model_def = functools.partial(get_model(model_name), config=config)
    input_shape = (
        jax.local_device_count(), config.data.image_size, config.data.image_size, config.data.num_channels)  # 3
    label_shape = input_shape[:1]
    fake_input = jnp.zeros(input_shape)
    fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
    params_rng, dropout_rng = jax.random.split(rng)
    model = model_def()
    deterministic_latent_input = not hasattr(config.model,
                                             'deterministic_latent_input') or config.model.deterministic_latent_input
    random_normal = None
    if not deterministic_latent_input:
        rng, step_rng = random.split(rng)
        random_normal = random.normal(step_rng, (fake_input.shape[0], config.model.latent_input_dim))

    variables = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_input, fake_label,
                           **({'x0': fake_input, 'random_normal': random_normal} if x0_input else {}))
    # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
    init_model_state, initial_params = variables.pop('params')
    return model, init_model_state, initial_params


def init_lambda_model(rng, config):
    """ Initialize a `flax.linen.Module` model. """
    model_name = 'lambdamodel'
    model_def = functools.partial(get_model(model_name), config=config)
    input_shape = (
        jax.local_device_count(), config.data.image_size, config.data.image_size, config.data.num_channels)  # 3
    label_shape = input_shape[:1]
    fake_input = jnp.zeros(input_shape)
    fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
    params_rng, dropout_rng = jax.random.split(rng)
    model = model_def()
    variables = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_input, fake_label)
    # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
    init_model_state, initial_params = variables.pop('params')
    return model, init_model_state, initial_params


def get_model_fn(model, params, states, train=False):
    # params = params_all['model']
    # states = states_all['model']
    """Create a function to give the output of the score-based model.

    Args:
      model: A `flax.linen.Module` object the represent the architecture of score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all mutable states.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels, rng=None, x0=None, z0=None, random_normal=None, use_mean=False, use_param_t=False):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
          rng: If present, it is the random state for dropout

        Returns:
          A tuple of (model output, new mutable states)
        """
        variables = {'params': params, **states}
        additional_params = {}
        if use_param_t:
            additional_params['use_param_t'] = use_param_t
        if not train:
            model_result, _ = model.apply(variables, x, labels, train=False, x0=x0, z0=z0, random_normal=random_normal,
                                          use_mean=use_mean, mutable=['batch_stats'], **additional_params)
            return model_result, states
        else:
            rngs = {'dropout': rng}
            outputs = model.apply(variables, x, labels, train=True, x0=x0, z0=z0, random_normal=random_normal,
                                  use_mean=use_mean, mutable=list(states.keys()),
                                  rngs=rngs, **additional_params)
            return outputs
            if states:
                return outputs
            else:
                return outputs, states

    return model_fn


def get_score_fn(sde, model, params, states, train=False, continuous=False, return_state=False, config=None):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all other mutable parameters.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.
      return_state: If `True`, return the new mutable states alongside the model output.

    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, params, states, train=train)
    use_param_t = config is not None and getattr(config.training, 'train_clf', False)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t, rng=None, x0=None, z0=None, random_normal=None, use_mean=False):
            # For VP-trained models, t=0 corresponds to the lowest noise level
            labels = t * (sde.N - 1)
            # Scale neural network output by standard deviation and flip sign
            assert(not use_param_t)
            model_result, state = model_fn(x, labels, rng, x0=x0, z0=z0, random_normal=random_normal, use_mean=use_mean,
                                           use_param_t=use_param_t)
            model = model_result['output']
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                std = sde.marginal_prob(jnp.zeros_like(x), t, params=params)[1]
            else:
                std = sde.sqrt_1m_alphas_cumprod[labels.astype(jnp.int32)]

            # TODO: DONE: scorescaling revert to normal
            model_result['score'] = batch_mul(-model, 1. / std)
            # TODO: DONE: meanprediction revert to normal
            # model_result['score'] = batch_mul(model - x, 1. / (std ** 2))

            # score = batch_mul(-model, 1. / std)
            if return_state:
                return model_result, state
            else:
                return model_result

    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t, rng=None, x0=None, z0=None, random_normal=None, use_mean=False):
            if continuous:
                labels = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = jnp.round(labels).astype(jnp.int32)

            additional_params = {}
            if use_param_t:
                additional_params['use_param_t'] = use_param_t
            model_result, state = model_fn(x, labels, rng, x0=x0, z0=z0, random_normal=random_normal, use_mean=use_mean,
                                           **additional_params)
            model = model_result['output']

            score = model

            # std = sde.marginal_prob(jnp.zeros_like(x), t, params=params)[1]
            # score = batch_mul(model, 1. / std)

            # if getattr(config.model, 'predict_x0', False):
            #    score = model - x
            # if getattr(config.model, 'predictx0', False):
            #    std = sde.marginal_prob(jnp.zeros_like(x), t, params=params)[1]
            #    model_result['score'] = batch_mul(model - x, 1. / (std ** 2))

            model_result['score'] = score
            if return_state:
                return model_result, state
            else:
                return model_result

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def to_flattened_numpy(x):
    """Flatten a JAX array `x` and convert it to numpy."""
    return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
    """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
    return jnp.asarray(x).reshape(shape)


def create_classifier(prng_key, batch_size, ckpt_path):
    """Create a noise-conditional image classifier.

    Args:
      prng_key: A JAX random state.
      batch_size: The batch size of input data.
      ckpt_path: The path to stored checkpoints for this classifier.

    Returns:
      classifier: A `flax.linen.Module` object that represents the architecture of the classifier.
      classifier_params: A dictionary that contains trainable parameters of the classifier.
    """
    input_shape = (batch_size, 32, 32, 3)
    classifier = wideresnet_noise_conditional.WideResnet(
        blocks_per_group=4,
        channel_multiplier=10,
        num_outputs=10
    )
    initial_variables = classifier.init({'params': prng_key, 'dropout': jax.random.PRNGKey(0)},
                                        jnp.ones(input_shape, dtype=jnp.float32),
                                        jnp.ones((batch_size,), dtype=jnp.float32), train=False)
    model_state, init_params = initial_variables.pop('params')
    classifier_params = checkpoints.restore_checkpoint(ckpt_path, init_params)
    return classifier, classifier_params


def get_logit_fn(classifier, classifier_params):
    """ Create a logit function for the classifier. """

    def preprocess(data):
        image_mean = jnp.asarray([[[0.49139968, 0.48215841, 0.44653091]]])
        image_std = jnp.asarray([[[0.24703223, 0.24348513, 0.26158784]]])
        return (data - image_mean[None, ...]) / image_std[None, ...]

    def logit_fn(data, ve_noise_scale):
        """Give the logits of the classifier.

        Args:
          data: A JAX array of the input.
          ve_noise_scale: time conditioning variables in the form of VE SDEs.

        Returns:
          logits: The logits given by the noise-conditional classifier.
        """
        data = preprocess(data)
        logits = classifier.apply({'params': classifier_params}, data, ve_noise_scale, train=False, mutable=False)
        return logits

    return logit_fn


def get_classifier_grad_fn(logit_fn):
    """Create the gradient function for the classifier in use of class-conditional sampling. """

    def grad_fn(data, ve_noise_scale, labels):
        def prob_fn(data):
            logits = logit_fn(data, ve_noise_scale)
            prob = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(labels.shape[0]), labels].sum()
            return prob

        return jax.grad(prob_fn)(data)

    return grad_fn
