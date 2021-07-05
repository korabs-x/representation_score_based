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
import jax
import functools
import jax.numpy as jnp
import numpy as np
import ml_collections

import sys
from typing import Any

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='cifarmodel')
class CifarModel(nn.Module):
    """NCSN++ model"""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, time_cond, train=True, x0=None, z0=None, random_normal=None, use_mean=False,
                 use_param_t=False):
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
        print(("num_resolutions", num_resolutions), file=sys.stderr)
        print(("ch_mult", ch_mult), file=sys.stderr)

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

        t_param = nn.sigmoid(self.param('param_t', lambda a, b: 0., 1))
        if use_param_t:
            std = config.model.sigma_min * (config.model.sigma_max / config.model.sigma_min) ** t_param
            time_cond = time_cond * 0. + std

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

        pyramid_downsample = None
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
            if x0 is not None:
                x0 = 2 * x0 - 1.

        z = False
        z_mean = False
        z_logvar = False
        clf_logits = False
        time_dependent_encoder = hasattr(config.model,
                                         'time_dependent_encoder') and config.model.time_dependent_encoder
        if getattr(config.model, 'include_latent_input', False):
            if x0 is None:
                assert z0 is not None
                input_shape = (
                    jax.local_device_count(), config.data.image_size, config.data.image_size,
                    config.data.num_channels)
                x0 = jnp.zeros(input_shape)

            latent_dim = config.model.latent_input_dim
            deterministic_latent_input = not hasattr(config.model,
                                                     'deterministic_latent_input') or config.model.deterministic_latent_input
            temb_enc = temb if time_dependent_encoder else None
            z0x0 = Encoder(nf=nf,
                           ch_mult=ch_mult,
                           num_res_blocks=num_res_blocks,
                           num_resolutions=num_resolutions,
                           ResnetBlock=ResnetBlock,
                           AttnBlock=AttnBlock,
                           attn_resolutions=attn_resolutions,
                           resblock_type=resblock_type,
                           progressive_input=progressive_input,
                           pyramid_downsample=pyramid_downsample,
                           combiner=combiner,
                           skip_rescale=skip_rescale,
                           Downsample=Downsample,
                           act=act,
                           latent_dim=latent_dim,
                           deterministic_latent_input=deterministic_latent_input)(x0, train=train, temb=temb_enc)

            z_mean = z0x0
            z_logvar = z0x0
            if not deterministic_latent_input and z0 is None:
                assert random_normal is not None or use_mean is not None
                z_mean = z0x0[:, :latent_dim]
                z_logvar = z0x0[:, latent_dim:] - 5.
                z_std = jnp.exp(0.5 * z_logvar)
                z0x0 = z_mean
                if use_mean is None or not use_mean:
                    z0x0 += random_normal * z_std

            if z0 is None:
                z0 = z0x0
            z = z0

            # classifier
            z_clf = z
            z_clf = act(nn.Dense(nf * 2)(z_clf))
            z_clf = act(nn.Dense(nf * 2)(z_clf))
            z_clf = act(nn.Dense(nf)(z_clf))
            z_clf = nn.Dense(10)(z_clf)
            clf_logits = nn.log_softmax(z_clf)

            # initial decode steps
            z0_dec = z0
            z0_dec = act(nn.Dense(nf * (1 if latent_dim < nf else 2))(z0_dec))
            z0_dec = act(nn.Dense(nf * 2)(z0_dec))
            z0_dec = act(nn.Dense(nf * 4)(z0_dec))
            if time_dependent_encoder:
                z0_dec = jnp.concatenate([act(temb), z0_dec], axis=-1)
                z0_dec = act(nn.Dense(nf * 4)(z0_dec))
            z0_dec = nn.Dense(nf * 4)(z0_dec)
            temb = jnp.concatenate([temb, z0_dec], axis=-1)

        # Downsampling block
        input_pyramid = None
        if progressive_input != 'none':
            input_pyramid = x

        # print(("Input", x.shape), file=sys.stderr)
        hs = [conv3x3(x, nf)]
        # print(("conv3x3", hs[-1].shape), file=sys.stderr)
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
                # print(("ResnetBlock", h.shape), file=sys.stderr)
                if h.shape[1] in attn_resolutions:
                    h = AttnBlock()(h)
                    # print(("AttnBlock", h.shape), file=sys.stderr)
                hs.append(h)

            if i_level != num_resolutions - 1:
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

        smallest_h_shape = h.shape
        if getattr(config.model, 'include_latent_input', False):
            z0_dec_late = z0_dec
            if True:
                z0_dec_late = act(nn.Dense(nf * 4)(act(z0_dec_late)))
                z0_dec_late = act(nn.Dense(nf * 8)(z0_dec_late))
                z0_dec_late = act(nn.Dense(nf * 16)(z0_dec_late))
                if time_dependent_encoder:
                    z0_dec_late = jnp.concatenate([act(temb), z0_dec_late], axis=-1)
                z0_dec_late = act(nn.Dense(nf * 32)(z0_dec_late))
                z0_dec_late = act(nn.Dense(nf * 64)(z0_dec_late))
                z0_dec_late = act(nn.Dense(h.shape[-1] * h.shape[-2] * h.shape[-3])(z0_dec_late))
                z0_dec_late = jnp.reshape(z0_dec_late, h.shape)
                h = jnp.concatenate([h, z0_dec_late], axis=-1)

        h = ResnetBlock()(h, temb, train)
        # print(("ResnetBlock", h.shape), file=sys.stderr)
        h = AttnBlock()(h)
        # print(("AttnBlock", h.shape), file=sys.stderr)
        h = ResnetBlock()(h, temb, train)
        # print(("ResnetBlock", h.shape), file=sys.stderr)

        pyramid = None

        # Upsampling block
        # print(("----- Start Upsampling -----"), file=sys.stderr)
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

            if i_level != 0:
                if resblock_type == 'ddpm':
                    h = Upsample()(h)
                    # print(("Upsample", h.shape), file=sys.stderr)
                else:
                    h = ResnetBlock(up=True)(h, temb, train)
                    # print(("ResnetBlock", h.shape), file=sys.stderr)

        assert not hs

        if progressive == 'output_skip':
            h = pyramid
        else:
            h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
            # print(("act", h.shape), file=sys.stderr)
            h = conv3x3(h, x.shape[-1], init_scale=init_scale)
            # print(("conv3x3", h.shape), file=sys.stderr)

        if config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        reconstruction = False
        lambda_reconstr = getattr(config.model, 'lambda_reconstr', 0.0)
        lambda_reconstr_rate = getattr(config.model, 'lambda_reconstr_rate', 0.0)
        if lambda_reconstr != 0. or lambda_reconstr_rate != 0.:
            assert (getattr(config.model, 'include_latent_input', False))
            # reconstruct x0 from latent
            reconstruction = Decoder(nf=nf,
                                     ch_mult=ch_mult,
                                     num_res_blocks=num_res_blocks,
                                     num_resolutions=num_resolutions,
                                     ResnetBlock=ResnetBlock,
                                     AttnBlock=AttnBlock,
                                     attn_resolutions=attn_resolutions,
                                     resblock_type=resblock_type,
                                     progressive_input=progressive_input,
                                     progressive=progressive,
                                     pyramid_downsample=pyramid_downsample,
                                     combiner=combiner,
                                     skip_rescale=skip_rescale,
                                     Downsample=Downsample,
                                     act=act,
                                     smallest_h_shape=smallest_h_shape,
                                     x_shape=x.shape,
                                     init_scale=init_scale,
                                     config=self.config)(z0, train=train, temb=temb,
                                                         time_dependent_encoder=time_dependent_encoder)

        # print(("Finished", h.shape), file=sys.stderr)
        # raise Exception('Test')
        return {'output': h,
                'latent': z,
                'clf_logits': clf_logits,
                'z_mean': z_mean,
                'z_logvar': z_logvar,
                'reconstruction': reconstruction}


class Encoder(nn.Module):
    nf: Any
    num_resolutions: Any
    num_res_blocks: Any
    ch_mult: Any
    attn_resolutions: Any
    ResnetBlock: Any
    AttnBlock: Any
    act: Any
    resblock_type: Any
    progressive_input: Any
    pyramid_downsample: Any
    combiner: Any
    skip_rescale: Any
    Downsample: Any
    latent_dim: Any
    deterministic_latent_input: Any

    @nn.compact
    def __call__(self, x0, train=True, temb=None):
        # temb = None
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x0

        hs = [conv3x3(x0, self.nf)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.ResnetBlock(out_ch=self.nf * self.ch_mult[i_level])(hs[-1], temb, train)
                if h.shape[1] in self.attn_resolutions:
                    h = self.AttnBlock()(h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = self.Downsample()(hs[-1])
                else:
                    h = self.ResnetBlock(down=True)(hs[-1], temb, train)

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample()(input_pyramid)
                    h = self.combiner()(input_pyramid, h)

                elif self.progressive_input == 'residual':
                    input_pyramid = self.pyramid_downsample(out_ch=h.shape[-1])(input_pyramid)
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = self.ResnetBlock()(h, temb, train)
        h = self.AttnBlock()(h)
        h = self.ResnetBlock()(h, temb, train)

        z0x0 = h
        z0x0 = z0x0.reshape((z0x0.shape[0], -1))
        z0x0 = nn.Dense(self.nf * 2)(self.act(z0x0))
        z0x0 = nn.Dense(self.nf * (1 if self.latent_dim < self.nf and self.deterministic_latent_input else 2))(
            self.act(z0x0))
        z0x0 = nn.Dense(self.latent_dim * (1 if self.deterministic_latent_input else 2))(self.act(z0x0))

        return z0x0


class Decoder(nn.Module):
    nf: Any
    num_resolutions: Any
    num_res_blocks: Any
    ch_mult: Any
    attn_resolutions: Any
    ResnetBlock: Any
    AttnBlock: Any
    act: Any
    resblock_type: Any
    progressive_input: Any
    progressive: Any
    pyramid_downsample: Any
    combiner: Any
    skip_rescale: Any
    Downsample: Any
    smallest_h_shape: Any
    x_shape: Any
    init_scale: Any
    config: Any

    @nn.compact
    def __call__(self, z, train=True, temb=None, time_dependent_encoder=False):
        if not time_dependent_encoder:
            temb = None

        z0_dec = z
        z0_dec = self.act(nn.Dense(self.nf * 2)(z0_dec))
        z0_dec = self.act(nn.Dense(self.nf * 4)(z0_dec))
        if time_dependent_encoder:
            z0_dec = jnp.concatenate([self.act(temb), z0_dec], axis=-1)
            z0_dec = self.act(nn.Dense(self.nf * 4)(z0_dec))
        z0_dec = nn.Dense(self.nf * 4)(z0_dec)

        z0_dec_late = self.act(z0_dec)
        z0_dec_late = self.act(nn.Dense(self.nf * 4)(z0_dec_late))
        z0_dec_late = self.act(nn.Dense(self.nf * 8)(z0_dec_late))
        z0_dec_late = self.act(nn.Dense(self.nf * 16)(z0_dec_late))
        if time_dependent_encoder:
            z0_dec_late = jnp.concatenate([self.act(temb), z0_dec_late], axis=-1)
        z0_dec_late = self.act(nn.Dense(self.nf * 32)(z0_dec_late))
        z0_dec_late = self.act(nn.Dense(self.nf * 64)(z0_dec_late))
        z0_dec_late = self.act(
            nn.Dense(self.smallest_h_shape[-1] * self.smallest_h_shape[-2] * self.smallest_h_shape[-3])(z0_dec_late))
        z0_dec_late = jnp.reshape(z0_dec_late, self.smallest_h_shape)
        temb = jnp.concatenate([temb, z0_dec], axis=-1) if time_dependent_encoder else z0_dec

        h = z0_dec_late

        assert self.progressive == 'none'

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.ResnetBlock(out_ch=self.nf * self.ch_mult[i_level])(h, temb, train)
            if h.shape[1] in self.attn_resolutions:
                h = self.AttnBlock()(h)

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = self.Upsample()(h)
                else:
                    h = self.ResnetBlock(up=True)(h, temb, train)

        h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
        h = conv3x3(h, self.x_shape[-1], init_scale=self.init_scale)

        return h


class SmallDecoder(nn.Module):

    @nn.compact
    def __call__(self, z):
        hid_ch = 32
        kernel_size = (4, 4)
        hidden_dim = 256
        reshape = (*kernel_size, hid_ch)
        cnn_kwargs = dict(strides=(2, 2))  # , padding=((1, 1), (1, 1))

        h = z
        h = nn.Dense(hidden_dim)(h)
        h = nn.swish(h)
        h = nn.Dense(hidden_dim)(h)
        h = nn.swish(h)
        h = nn.Dense(np.prod(reshape))(h)
        h = nn.swish(h)
        h = h.reshape(h.shape[0], *reshape)
        h = nn.ConvTranspose(hid_ch, kernel_size, **cnn_kwargs)(h)
        h = nn.swish(h)
        h = nn.ConvTranspose(hid_ch, kernel_size, **cnn_kwargs)(h)
        h = nn.swish(h)
        h = nn.ConvTranspose(hid_ch, kernel_size, **cnn_kwargs)(h)
        h = nn.swish(h)
        h = nn.Conv(3, kernel_size, padding='SAME')(h)
        # h = nn.sigmoid(h)
        return h
