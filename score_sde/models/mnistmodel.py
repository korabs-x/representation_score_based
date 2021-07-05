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
import numpy as np
import ml_collections
import jax.nn as jnn

import sys

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='mnistmodel')
class MnistModel(nn.Module):
    """Mnist model"""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, time_cond, train=True, x0=None, z0=None):
        # config parsing
        config = self.config
        act = get_act(config)

        dropout = config.model.dropout
        nf = 64

        embedding_type = config.model.embedding_type.lower()
        assert embedding_type in ['fourier', 'positional']

        if embedding_type == 'positional':
            timesteps = time_cond
            temb = layers.get_timestep_embedding(timesteps, nf)
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        temb = act(nn.Dense(nf * 4, kernel_init=default_initializer())(temb))
        temb = act(nn.Dense(nf * 4, kernel_init=default_initializer())(temb))

        if not config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # """
        # really simple MLP network
        h = x
        h = nn.Conv(nf, kernel_size=(4, 4), padding='SAME', strides=(2, 2))(h)
        h += nn.Dense(nf)(temb)[:, None, None, :]
        h = act(h)
        h = nn.Conv(nf * 2, kernel_size=(4, 4), padding='SAME', strides=(2, 2))(h)
        h += nn.Dense(nf * 2)(temb)[:, None, None, :]
        h = act(h)
        h = nn.BatchNorm()(h)
        h = nn.ConvTranspose(nf, kernel_size=(4, 4), padding='SAME', strides=(2, 2))(h)
        h += nn.Dense(nf)(temb)[:, None, None, :]
        h = act(h)
        h = nn.ConvTranspose(1, kernel_size=(4, 4), padding='SAME', strides=(2, 2))(h)
        z = h * 0
        return {'output': h, 'latent': z}

        h = h.reshape(h.shape[0], 28 * 28)
        h = jnp.concatenate([h, temb], axis=-1)
        h = act(nn.Dense(1024)(h))
        h = act(nn.Dense(1024)(h))
        h = act(nn.Dense(1024)(h))
        h = act(nn.Dense(1024)(h))
        h = act(nn.Dense(1024)(h))
        z = h * 0
        h = nn.Dense(28 * 28)(h)
        h = h.reshape((h.shape[0], 28, 28, 1))
        return {'output': h, 'latent': z}
        # """

        # Downsampling block
        h = x

        h = nn.Conv(nf, kernel_size=(4, 4), padding='SAME', strides=(2, 2))(h)
        # print(("Conv", h.shape), file=sys.stderr)
        h += nn.Dense(nf)(temb)[:, None, None, :]
        h = act(h)

        h = nn.Conv(nf * 2, kernel_size=(4, 4), padding='SAME', strides=(2, 2))(h)
        # print(("Conv", h.shape), file=sys.stderr)
        h += nn.Dense(nf * 2)(temb)[:, None, None, :]
        h = act(h)
        # h = nn.BatchNorm()(h)

        """

        h = jnp.reshape(h, (h.shape[0], -1))

        h = nn.Dense(1024)(h)
        h = act(h)
        h = nn.Dense(128)(h)
        h = act(h)
        # h = nn.BatchNorm()(h)
        h = nn.Dense(10)(h)

        z = h

        # Upsampling block
        h = nn.Dense(1024)(h)
        h = act(h)
        # h = nn.BatchNorm()(h)
        h = nn.Dense(7 * 7 * nf * 2)(h)
        h = jnp.reshape(h, (h.shape[0], 7, 7, nf * 2))
        h += nn.Dense(nf * 2)(temb)[:, None, None, :]
        h = act(h)
        # h = nn.BatchNorm()(h)
        """
        z = h * 0

        h = nn.ConvTranspose(nf, kernel_size=(4, 4), padding='SAME', strides=(2, 2))(h)
        # print(("ConvTranspose", h.shape), file=sys.stderr)
        h += nn.Dense(nf)(temb)[:, None, None, :]
        h = act(h)
        # h = nn.BatchNorm()(h)
        h = nn.ConvTranspose(1, kernel_size=(4, 4), padding='SAME', strides=(2, 2))(h)
        # print(("ConvTranspose", h.shape), file=sys.stderr)

        return {'output': h, 'latent': z}
