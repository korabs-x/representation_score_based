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

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='toymodel')
class ToyModel(nn.Module):
    """NCSN++ model"""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, time_cond, train=True, x0=None, z0=None, random_normal=None, use_mean=False):
        # config parsing
        config = self.config
        act = get_act(config)
        sigmas = utils.get_sigmas(config)

        nf = config.model.nf
        ch_mult = config.model.ch_mult
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        num_resolutions = len(ch_mult)

        conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        skip_rescale = config.model.skip_rescale
        resblock_type = config.model.resblock_type.lower()
        progressive = config.model.progressive.lower()
        progressive_input = config.model.progressive_input.lower()
        embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        # SDE parameters
        # self.param('sde_beta_min', lambda a, b: config.model.beta_min, 1)
        self.param('sde_beta_min', lambda a, b: config.model.beta_min, 1)
        self.param('sde_beta_max', lambda a, b: config.model.beta_max, 1)
        self.param('sde_beta', lambda a, b: config.model.beta_max * 0.6, 1)
        # self.param('sde_loss_weight', lambda a, b: config.model.beta_max, 1)

        # timestep/noise_level embedding; only for continuous training
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
            used_sigmas = sigmas[time_cond.astype(jnp.int32)]
            temb = layers.get_timestep_embedding(timesteps, nf)
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
            temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))
        else:
            temb = None

        # flatten data
        x_flat = jnp.reshape(x, (x.shape[0], -1))
        h = x_flat
        h = nn.Dense(nf * 4, kernel_init=default_initializer())(h)

        z_mean, z_logvar = False, False
        if hasattr(config.model, 'include_latent_input') and config.model.include_latent_input:
            if x0 is None:
                assert z0 is not None
                input_shape = (
                    jax.local_device_count(), config.data.image_size, config.data.image_size, config.data.num_channels)
                x0 = jnp.zeros(input_shape)

            latent_dim = config.model.latent_input_dim

            # if z0 is None and x0 is not None:
            # always push some data through these layers to trigger correct naming of following layers
            deterministic_latent_input = not hasattr(config.model,
                                                     'deterministic_latent_input') or config.model.deterministic_latent_input

            x0_flat = jnp.reshape(x0, (x0.shape[0], -1))
            z0_x0 = x0_flat
            if hasattr(config.model, 'affine_latent_input') and config.model.affine_latent_input:
                z0_x0 = nn.Dense(latent_dim, kernel_init=default_initializer())(z0_x0)
            else:
                z0_x0 = nn.Dense(1024, kernel_init=default_initializer())(z0_x0)
                z0_x0 = act(z0_x0)
                z0_x0 = nn.Dense(1024, kernel_init=default_initializer())(z0_x0)
                z0_x0 = act(z0_x0)
                z0_x0 = nn.Dense(1024, kernel_init=default_initializer())(z0_x0)
                z0_x0 = act(z0_x0)
                z0_x0 = nn.Dense(latent_dim * (1 if deterministic_latent_input else 2), kernel_init=default_initializer())(z0_x0)

            if not deterministic_latent_input:
                assert random_normal is not None or use_mean is not None
                z_mean = z0_x0[:, :latent_dim]
                z_logvar = z0_x0[:, latent_dim:] - 5.
                z_std = jnp.exp(0.5 * z_logvar)
                z0_x0 = z_mean
                if use_mean is None or not use_mean:
                    z0_x0 += random_normal * z_std

            if z0 is None:
                z0 = z0_x0

            z = z0

            if z0 is not None:
                z0 = nn.Dense(1024, kernel_init=default_initializer())(z0)
                z0 = act(z0)
                z0 = nn.Dense(1024, kernel_init=default_initializer())(z0)
                z0 = act(z0)
                z0 = nn.Dense(nf * 4, kernel_init=default_initializer())(z0)
        else:
            z0 = None
            z = h * 0.

        #if z_mean is False:
        #    raise Exception((deterministic_latent_input, x0 is None, z0 is None, random_normal is None))

        # concatenate with time input
        h = jnp.concatenate([act(temb), act(h)] + ([] if z0 is None else [act(z0)]), axis=-1)

        h = nn.Dense(1024, kernel_init=default_initializer())(h)
        h = act(h)
        h = nn.Dense(1024, kernel_init=default_initializer())(h)
        h = act(h)
        h = nn.Dense(1024, kernel_init=default_initializer())(h)
        h = act(h)
        h = nn.Dense(1024, kernel_init=default_initializer())(h)
        h = act(h)
        h = nn.Dense(1024, kernel_init=default_initializer())(h)
        h = act(h)
        h = nn.Dense(256, kernel_init=default_initializer())(h)
        h = act(h)
        h = nn.Dense(64, kernel_init=default_initializer())(h)
        h = act(h)

        num_ch = config.data.num_channels
        h = nn.Dense(num_ch, kernel_init=default_initializer())(h)

        h = jnp.reshape(h, (h.shape[0], 1, 1, num_ch))

        print(("Finished", h.shape), file=sys.stderr)
        if False and h.shape[0] > 1:
            raise Exception(str(x.shape) + ' - ' + str(h.shape))
        return {'output': h, 'latent': z, 'z_mean': z_mean, 'z_logvar': z_logvar}
