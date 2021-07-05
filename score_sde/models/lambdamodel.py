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

from . import utils, layers, layerspp, normalization
import flax.linen as nn
import functools
import jax.numpy as jnp
import jax
import numpy as np
import ml_collections

import sys

get_act = layers.get_act
default_initializer = layers.default_init


@utils.register_model(name='lambdamodel')
class LambdaModel(nn.Module):
    """NCSN++ model"""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, time_cond, train=True, x0=None, z0=None, random_normal=None, use_mean=None):
        # register parameters
        # lambda_scalar = self.param('lambda_scalar', lambda a, b: 1., 1)
        # h = time_cond * 0. + lambda_scalar

        x = x.reshape(x.shape[0], -1)

        lm_type = getattr(self.config.training, 'lm_type', '')
        lm_input = getattr(self.config.training, 'lm_input', 't')
        use_mog = lm_type == 'mog'
        only_time_dependent = lm_input == 't'

        means, stds = None, None
        if lm_type == 'linear':
            a = nn.sigmoid(self.param('a', lambda a, b: 0., 1)) * 2 - 1
            t = time_cond * 1. / 1000.
            h = 2 * a * t + (1 - a)

        elif lm_type == 'powera':
            a = jnp.exp(self.param('a', lambda a, b: 0., 1))
            t = time_cond * 1. / 1000.
            h = (a + 1) * (1 - t) ** a

        elif lm_type == 'power2':
            a = nn.sigmoid(self.param('a', lambda a, b: 0., 1)) * 2
            t = time_cond * 1. / 1000.
            h = (a + 1) * (1 - t) ** a

        elif lm_type == 'power4':
            a = nn.sigmoid(self.param('a', lambda a, b: 0., 1)) * 4
            t = time_cond * 1. / 1000.
            h = (a + 1) * (1 - t) ** a

        elif lm_type == 'power16':
            a = nn.sigmoid(self.param('a', lambda a, b: 0., 1)) * 16
            t = time_cond * 1. / 1000.
            h = (a + 1) * (1 - t) ** a

        elif use_mog:
            if only_time_dependent:
                # x = jnp.concatenate((x, 4. * (time_cond * 1. / 1000. - .5).reshape(-1, 1)), axis=-1)
                # x = 4. * (time_cond * 1. / 1000. - .5).reshape(-1, 1)
                x = (time_cond * 1. / 1000 - 0.5).reshape(-1, 1)

            n_gaussians = getattr(self.config.training, 'lm_n_mog', 2)
            h = jnp.zeros(time_cond.shape)
            means = []
            stds = []
            for i in range(n_gaussians):
                """
                mean = self.param(f'lambda_mean_{i}',
                                  lambda a, b: jnp.concatenate(
                                      (jnp.zeros(shape=x.shape[-1] - 1),
                                       jnp.ones(shape=1) * 4. * ((i + 1) * 1. / (n_gaussians + 1) - .5))), 1)
                std = jnp.exp(
                    self.param(f'lambda_std_{i}', lambda a, b: jnp.log(jnp.ones(shape=x.shape[-1]) * 1.), 1)) + 1e-4
                """
                mean = self.param(f'lambda_mean_{i}', lambda a, b: jnp.ones(shape=x.shape[-1]) * 1. * (
                        (i + 1) * 1. / (n_gaussians + 1) - .5), 1)
                means.append(mean)
                std = jnp.exp(
                    self.param(f'lambda_std_{i}', lambda a, b: jnp.log(jnp.ones(shape=x.shape[-1]) * 2.), 1)) + 1e-4
                stds.append(std)

                # mean = self.param(f'lambda_mean_{i}', lambda a, b: jnp.ones(shape=x.shape[-1]) * (-2.), 1)
                # means.append(mean)
                # std = jnp.exp(
                #    self.param(f'lambda_std_{i}', lambda a, b: jnp.log(jnp.ones(shape=x.shape[-1]) * 1e-4), 1))#  + 1e-4
                # stds.append(std)
                p = 1. / (std * jnp.sqrt(2. * jnp.pi)) * jnp.exp(-0.5 * jnp.square((x - mean) / std))
                p = jnp.prod(p, axis=-1)
                h += 1. / n_gaussians * p

            h *= 2
            # h /= jnp.mean(h)

        else:

            config = self.config
            embedding_type = config.model.embedding_type.lower()
            act = get_act(config)
            nf = config.model.nf

            if embedding_type == 'fourier':
                # Gaussian Fourier features embeddings.
                assert config.training.continuous, "Fourier features are only used for continuous training."
                used_sigmas = time_cond
                temb = layerspp.GaussianFourierProjection(
                    embedding_size=nf,
                    scale=config.model.fourier_scale)(jnp.log(used_sigmas))

            elif embedding_type == 'positional':
                # Sinusoidal positional embeddings.
                timesteps = time_cond
                temb = layers.get_timestep_embedding(timesteps, nf)
            else:
                raise ValueError(f'embedding type {embedding_type} unknown.')

            temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
            temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))

            h = x
            h = nn.Dense(nf * 4, kernel_init=default_initializer())(h)

            # concatenate with time input
            h = jnp.concatenate([act(temb), act(h)], axis=-1)
            h = nn.Dense(1024, kernel_init=default_initializer())(h)
            h = act(h)
            h = nn.Dense(1024, kernel_init=default_initializer())(h)
            h = act(h)
            h = nn.Dense(1024, kernel_init=default_initializer())(h)
            h = act(h)
            h = nn.Dense(1024, kernel_init=default_initializer())(h)
            h = act(h)
            h = nn.Dense(1, kernel_init=default_initializer())(h)
            h = jnp.exp(h)
            # h = nn.BatchNorm()(h)

            h = h.reshape(time_cond.shape)
            z = h

            apply_norm = False
            # if h.shape[0] > 1:
            # raise Exception((h.shape, h_mean.shape, h_mean))
            if apply_norm:
                h_mean = h.mean(axis=(0,))
                h = h * 1. / h_mean.clip(a_min=1e-1)
            # h *= 10

        return {'lambda': h, 'latent': h, 'means': means, 'stds': stds}
