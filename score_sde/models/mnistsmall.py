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
from typing import Any
import flax.linen as nn
import functools
import jax.numpy as jnp
import numpy as np
import ml_collections
import jax

import sys

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='mnistsmall')
class MnistSmall(nn.Module):
    """NCSN++ model"""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, time_cond, train=True, x0=None, z0=None, random_normal=None, use_mean=None):
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

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        if not config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        input_pyramid = None
        if progressive_input != 'none':
            input_pyramid = x

        # Add custom latent space
        z = False
        z_mean = False
        z_logvar = False
        if hasattr(config.model, 'include_latent_input') and config.model.include_latent_input:
            if x0 is None:
                assert z0 is not None
                input_shape = (
                    jax.local_device_count(), config.data.image_size, config.data.image_size,
                    config.data.num_channels)
                x0 = jnp.zeros(input_shape)

            latent_dim = config.model.latent_input_dim
            deterministic_latent_input = not hasattr(config.model,
                                                     'deterministic_latent_input') or config.model.deterministic_latent_input

            use_encoder = True

            if use_encoder:
                time_dependent_encoder = hasattr(config.model, 'time_dependent_encoder') and config.model.time_dependent_encoder
                temb_enc = temb if time_dependent_encoder else None
                z0x0 = Encoder(nf=nf,
                               ch_mult=ch_mult,
                               num_res_blocks=num_res_blocks,
                               ResnetBlock=ResnetBlock,
                               AttnBlock=AttnBlock,
                               act=act,
                               latent_dim=latent_dim,
                               deterministic_latent_input=deterministic_latent_input)(x0, train=train, temb=temb_enc)
            else:
                z0x0 = conv3x3(x0, nf)

                ch_mult_x0 = ch_mult
                num_resolutions_x0 = len(ch_mult_x0)
                num_res_blocks_x0 = num_res_blocks
                for i_level in range(num_resolutions_x0):
                    for i_block in range(num_res_blocks_x0):
                        z0x0 = ResnetBlock(out_ch=nf * ch_mult_x0[i_level])(z0x0, None, train)
                        # print(("z0x0 ResnetBlock", z0x0.shape), file=sys.stderr)
                    if i_level < num_resolutions_x0 - 1:
                        z0x0 = ResnetBlock(down=True)(z0x0, None, train)
                        # print(("z0x0 ResnetBlock down", z0x0.shape), file=sys.stderr)
                z0x0 = ResnetBlock(out_ch=nf)(z0x0, None, train)
                # print(("z0x0 ResnetBlock out_ch", z0x0.shape), file=sys.stderr)
                z0x0 = AttnBlock()(z0x0)
                # print(("z0x0 AttnBlock", z0x0.shape), file=sys.stderr)
                z0x0 = z0x0.reshape((z0x0.shape[0], -1))
                # z0x0 = jnp.squeeze(z0x0, axis=(1, 2))
                z0x0 = nn.Dense(nf * 2)(act(z0x0))
                z0x0 = nn.Dense(nf * (1 if latent_dim < nf and deterministic_latent_input else 2))(
                    act(z0x0))
                z0x0 = nn.Dense(latent_dim * (1 if deterministic_latent_input else 2))(act(z0x0))

            z_mean = z0x0
            z_logvar = z0x0
            if not deterministic_latent_input and z0 is None:
                assert random_normal is not None or use_mean is not None
                z_mean = z0x0[:, :latent_dim]
                z_logvar = z0x0[:, latent_dim:] - 5.
                # z0x0 = z0x0[:, :latent_dim]
                z_std = jnp.exp(0.5 * z_logvar)
                z0x0 = z_mean
                if use_mean is None or not use_mean:
                    z0x0 += random_normal * z_std

            if z0 is None:
                z0 = z0x0
            z = z0

            z0_dec = z0
            z0_dec = act(nn.Dense(nf * (1 if latent_dim < nf else 2))(z0_dec))
            z0_dec = act(nn.Dense(nf * 2)(z0_dec))
            z0_dec = act(nn.Dense(nf * 2)(z0_dec))
            z0_dec = nn.Dense(nf * 4)(z0_dec)
            temb = jnp.concatenate([temb, z0_dec], axis=-1)
            # temb = nn.Dense(nf * 8)(act(temb))
            # temb = nn.Dense(nf * 8)(act(temb))
            # h += z0_dec[:, None, None, :]
            # h = jnp.concatenate([h, h * 0. + z0_dec[:, None, None, :]], axis=-1)

        # Downsampling block
        # print(("Input", x.shape), file=sys.stderr)
        h = conv3x3(x, nf)
        # print(("conv3x3", h.shape), file=sys.stderr)
        # h = conv3x3(act(h), nf, padding='VALID')
        # h = conv3x3(act(h), nf, padding='VALID')
        hs = [h]
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
                # print(("ResnetBlock", h.shape), file=sys.stderr)
                if h.shape[1] in attn_resolutions:
                    h = AttnBlock()(h)
                    # print(("AttnBlock", h.shape), file=sys.stderr)
                hs.append(h)

            if i_level < num_resolutions - 1:
                if resblock_type == 'ddpm':
                    h = Downsample()(hs[-1])
                    # print(("Downsample", h.shape), file=sys.stderr)
                else:
                    # MNIST problem downsample / downsampling being tried from shape 7 by factor 2
                    h = ResnetBlock(down=True)(hs[-1], temb, train)
                    # print(("ResnetBlock down", h.shape), file=sys.stderr)

                if progressive_input == 'input_skip':
                    input_pyramid = pyramid_downsample()(input_pyramid)
                    h = combiner()(input_pyramid, h)
                    # print(("pyramid_downsample+combiner", h.shape), file=sys.stderr)

                elif progressive_input == 'residual':
                    input_pyramid = pyramid_downsample(out_ch=h.shape[-1])(input_pyramid)
                    # print(("NO CALL EXPECTED 2", h.shape), file=sys.stderr)
                    if skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                        # print(("NO CALL EXPECTED 2.1", input_pyramid.shape), file=sys.stderr)
                    else:
                        input_pyramid = input_pyramid + h
                        # print(("NO CALL EXPECTED 2.2", input_pyramid.shape), file=sys.stderr)
                    h = input_pyramid
                    # print(("NO CALL EXPECTED 3", h.shape), file=sys.stderr)

                hs.append(h)

        h = hs[-1]

        if False:
            z0_dec_late = act(nn.Dense(nf * 4)(act(z0_dec)))
            z0_dec_late = act(nn.Dense(nf * 8)(z0_dec_late))
            z0_dec_late = act(nn.Dense(nf * 16)(z0_dec_late))
            z0_dec_late = act(nn.Dense(nf * 32)(z0_dec_late))
            z0_dec_late = act(nn.Dense(nf * 64)(z0_dec_late))
            z0_dec_late = act(nn.Dense(h.shape[-1] * h.shape[-2] * h.shape[-3])(z0_dec_late))
            z0_dec_late = jnp.reshape(z0_dec_late, h.shape)
            h = jnp.concatenate([h, z0_dec_late], axis=-1)

        if hasattr(config.model, 'include_latent_input') and config.model.include_latent_input:
            h = ResnetBlock()(h, z0_dec, train)
            h = AttnBlock()(h)
            h = ResnetBlock()(h, z0_dec, train)

        h = ResnetBlock()(h, temb, train)
        # print(("ResnetBlock", h.shape), file=sys.stderr)
        h = AttnBlock()(h)
        # print(("AttnBlock", h.shape), file=sys.stderr)
        h = ResnetBlock()(h, temb, train)
        # print(("ResnetBlock", h.shape), file=sys.stderr)

        pyramid = None

        # Upsampling block
        # print("----- Start Upsampling -----", file=sys.stderr)
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(jnp.concatenate([h, hs.pop()], axis=-1),
                                                              temb,
                                                              train)
                # print(("ResnetBlock", h.shape), file=sys.stderr)

            if h.shape[1] in attn_resolutions:
                h = AttnBlock()(h)
                # print(("AttnBlock", h.shape), file=sys.stderr)

            if progressive != 'none':
                # print(("NO CALL EXPECTED 4", h.shape), file=sys.stderr)
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        pyramid = conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            x.shape[-1],
                            bias=True,
                            init_scale=init_scale)
                    elif progressive == 'residual':
                        pyramid = conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            h.shape[-1],
                            bias=True)
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        pyramid = pyramid_upsample()(pyramid)
                        pyramid = pyramid + conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            x.shape[-1],
                            bias=True,
                            init_scale=init_scale)
                    elif progressive == 'residual':
                        pyramid = pyramid_upsample(out_ch=h.shape[-1])(pyramid)
                        if skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level > 0:
                if resblock_type == 'ddpm':
                    h = Upsample()(h)
                    # print(("Upsample", h.shape), file=sys.stderr)
                else:
                    h = ResnetBlock(up=True)(h, temb, train)
                    # print(("ResnetBlock", h.shape), file=sys.stderr)

        # assert not hs

        if progressive == 'output_skip':
            h = pyramid
        else:
            h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
            # print(("GroupNorm", h.shape), file=sys.stderr)
            h = conv3x3(h, x.shape[-1], init_scale=init_scale)
            # print(("conv3x3", h.shape), file=sys.stderr)

        if config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        # print(("Finished", h.shape), file=sys.stderr)
        # raise Exception('Test')
        return {'output': h, 'latent': z, 'z_mean': z_mean, 'z_logvar': z_logvar}


class Encoder(nn.Module):
    nf: Any
    ch_mult: Any
    num_res_blocks: Any
    ResnetBlock: Any
    AttnBlock: Any
    act: Any
    latent_dim: Any
    deterministic_latent_input: Any

    @nn.compact
    def __call__(self, x0, train=True, temb=None):
        z0x0 = conv3x3(x0, self.nf)

        ch_mult_x0 = self.ch_mult
        num_resolutions_x0 = len(ch_mult_x0)
        num_res_blocks_x0 = self.num_res_blocks
        for i_level in range(num_resolutions_x0):
            for i_block in range(num_res_blocks_x0):
                z0x0 = self.ResnetBlock(out_ch=self.nf * ch_mult_x0[i_level])(z0x0, temb, train)
                # print(("z0x0 ResnetBlock", z0x0.shape), file=sys.stderr)
            if i_level < num_resolutions_x0 - 1:
                z0x0 = self.ResnetBlock(down=True)(z0x0, temb, train)
                # print(("z0x0 ResnetBlock down", z0x0.shape), file=sys.stderr)
        z0x0 = self.ResnetBlock(out_ch=self.nf)(z0x0, temb, train)
        # print(("z0x0 ResnetBlock out_ch", z0x0.shape), file=sys.stderr)
        z0x0 = self.AttnBlock()(z0x0)
        # print(("z0x0 AttnBlock", z0x0.shape), file=sys.stderr)
        z0x0 = z0x0.reshape((z0x0.shape[0], -1))
        # z0x0 = jnp.squeeze(z0x0, axis=(1, 2))
        z0x0 = nn.Dense(self.nf * 2)(self.act(z0x0))
        z0x0 = nn.Dense(self.nf * (1 if self.latent_dim < self.nf and self.deterministic_latent_input else 2))(
            self.act(z0x0))
        z0x0 = nn.Dense(self.latent_dim * (1 if self.deterministic_latent_input else 2))(self.act(z0x0))
        return z0x0
