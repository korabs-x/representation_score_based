# Copyright 2021 The Flax Authors.
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

# See issue #620.
# pytype: disable=attribute-error
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args

import math
from PIL import Image

from absl import app
from absl import flags
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.config import config
from flax import linen as nn
from flax import optim
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import checkpoints
import logging


import sys

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=1e-3,
    help=('The learning rate for the Adam optimizer.')
)

flags.DEFINE_integer(
    'batch_size', default=128,
    help=('Batch size for training.')
)

flags.DEFINE_integer(
    'num_epochs', default=80,  # 80,
    help=('Number of training epochs.')
)

flags.DEFINE_integer(
    'latents', default=2,
    help=('Number of latent variables.')
)


class Encoder(nn.Module):
    latents: int

    @nn.compact
    def __call__(self, x):
        """
        x = nn.Dense(500, name='fc1')(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
        """
        hid_ch = 32
        kernel_size = (4, 4)
        hidden_dim = 256
        latent_dim = self.latents

        cnn_kwargs = dict(strides=(2, 2), padding=((1, 1), (1, 1)))
        h = x
        h = nn.Conv(hid_ch, kernel_size, **cnn_kwargs)(h)
        h = nn.swish(h)
        h = nn.Conv(hid_ch, kernel_size, **cnn_kwargs)(h)
        h = nn.swish(h)
        h = nn.Conv(hid_ch, kernel_size, **cnn_kwargs)(h)
        h = h.reshape(h.shape[0], -1)
        h = nn.swish(h)
        h = nn.Dense(hidden_dim)(h)
        h = nn.swish(h)
        h = nn.Dense(hidden_dim)(h)
        h = nn.swish(h)
        h = nn.Dense(latent_dim * 2)(h)

        mean_x = h[:, :latent_dim]
        logvar_x = h[:, latent_dim:]
        return mean_x, logvar_x


class Decoder(nn.Module):

    @nn.compact
    def __call__(self, z):
        """
        z = nn.Dense(500, name='fc1')(z)
        z = nn.relu(z)
        z = nn.Dense(784, name='fc2')(z)
        """
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
        h = nn.ConvTranspose(hid_ch, kernel_size, padding=((1, 1), (1, 1)), **cnn_kwargs)(h)
        h = nn.swish(h)
        h = nn.ConvTranspose(hid_ch, kernel_size, **cnn_kwargs)(h)
        h = nn.swish(h)
        h = nn.Conv(1, kernel_size, padding='SAME')(h)

        # mean_x = nn.sigmoid(h[:, :, :, :1])
        # logvar_x = h[:, :, :, 1:]
        mean_x = nn.sigmoid(h)
        logvar_x = mean_x * 0. + self.param('logvar', lambda a, b: jnp.log(jnp.ones(shape=(1, *mean_x.shape[1:]))), 1)
        return mean_x, logvar_x


class VAE(nn.Module):
    latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_mean, recon_logvar = self.decoder(z)
        return recon_mean, recon_logvar, mean, logvar

    def generate(self, z, x_rng=None):
        x_mean, x_logvar = self.decoder(z)
        if x_rng is None:
            return x_mean
        return reparameterize(x_rng, x_mean, x_logvar)


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


@jax.vmap
def bpd(mean, logvar, x):
    sigma = jnp.exp(0.5 * logvar)
    # print((mean.shape, logvar.shape, x.shape), file=sys.stderr)
    # print(((x - mean) / sigma).shape, file=sys.stderr)
    # print((jnp.log(2 * jnp.pi * (sigma ** 2)) / 2).shape, file=sys.stderr)
    # print((jnp.square((x - mean) / sigma) / 2).shape, file=sys.stderr)
    # raise Exception()
    nll = jnp.mean(jnp.log(2 * jnp.pi * (sigma ** 2)) / 2 + jnp.square((x - mean) / sigma) / 2) / jnp.log(2.)

    #std = sigma
    #p = 1. / (std * jnp.sqrt(2. * jnp.pi)) * jnp.exp(-0.5 * jnp.square((x - mean) / std))
    #log_p = jnp.log(p)
    #nll = -jnp.mean(log_p)
    return nll


def compute_metrics(recon_x_mean, recon_x_logvar, x, mean, logvar):
    bce_loss = binary_cross_entropy_with_logits(recon_x_mean, x).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    bpd_sample = bpd(recon_x_mean, recon_x_logvar, x).mean()
    return {
        'bce': bce_loss,
        'kld': kld_loss,
        'bpd_sample': bpd_sample
    }


def model():
    return VAE(latents=FLAGS.latents)


@jax.jit
def train_step(optimizer, batch, z_rng):
    def loss_fn(params):
        recon_x_mean, recon_x_logvar, mean, logvar = model().apply({'params': params}, batch, z_rng)
        nll = bpd(recon_x_mean, recon_x_logvar, batch).mean()
        # bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = nll # + kld_loss
        return loss, recon_x_mean

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    _, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer


@jax.jit
def eval(params, batch, z, z_rng, z_rng2):
    def eval_model(vae):
        if batch is not None:
            images = batch  # ['image']
            recon_images_mean, recon_images_logvar, mean, logvar = vae(images, z_rng)
            recon_images_samples = reparameterize(z_rng, recon_images_mean, recon_images_logvar)
            recon_images_samples2 = reparameterize(z_rng2, recon_images_mean, recon_images_logvar)
            comparison = jnp.concatenate([images[:8].reshape(-1, 28, 28, 1),
                                          recon_images_mean[:8].reshape(-1, 28, 28, 1),
                                          recon_images_samples[:8].reshape(-1, 28, 28, 1),
                                          jnp.exp(recon_images_logvar[:8]).reshape(-1, 28, 28, 1),
                                          recon_images_samples2[:8].reshape(-1, 28, 28, 1)])

        generate_images = vae.generate(z, z_rng)
        generate_images = generate_images.reshape(-1, 28, 28, 1)
        if batch is not None:
            metrics = compute_metrics(recon_images_mean, recon_images_logvar, images, mean, logvar)
            return metrics, comparison, generate_images
        else:
            return generate_images

    return nn.apply(eval_model, model())({'params': params})


def prepare_image(x):
    # x['image'] = tf.cast(x['image'], tf.float32)
    # x = tf.reshape(x, (-1,))
    img = x['image']
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [28, 28], antialias=True)
    img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
    return {'image': img, 'label': x['label']}


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
    """Make a grid of images and Save it into an image file.
  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
    fp - A filename(string) or file object
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    scale_each (bool, optional): If ``True``, scale each image in the batch of
      images separately rather than the (min, max) over all images. Default: ``False``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the filename extension.
      If a file object was used instead of a filename, this parameter should always be used.
  """
    if not (isinstance(ndarray, jnp.ndarray) or
            (isinstance(ndarray, list) and all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError('array_like of tensors expected, got {}'.format(type(ndarray)))

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full((height * ymaps + padding, width * xmaps + padding, num_channels), pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = jax.ops.index_update(
                grid, jax.ops.index[y * height + padding:(y + 1) * height,
                      x * width + padding:(x + 1) * width],
                ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
    im = Image.fromarray(ndarr.copy())
    im.save(fp, format=format)


def main(argv):
    del argv

    #handler1 = logging.StreamHandler()
    # Make logging work for both disk and Google Cloud Storage
    #formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    #handler1.setFormatter(formatter)
    #logger = logging.getLogger()
    #logger.addHandler(handler1)
    #logger.setLevel('INFO')

    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

    rng = random.PRNGKey(0)
    rng, key = random.split(rng)

    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
    train_ds = train_ds.map(prepare_image)
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(60000)
    train_ds = train_ds.batch(FLAGS.batch_size)
    train_ds = iter(tfds.as_numpy(train_ds))

    test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
    test_ds = test_ds.map(prepare_image).batch(1000)
    test_ds = np.array(list(test_ds)[0]['image'])
    test_ds = jax.device_put(test_ds)

    init_data = jnp.ones((FLAGS.batch_size, 28, 28, 1), jnp.float32)
    params = model().init(key, init_data, rng)['params']

    optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(params)
    optimizer = jax.device_put(optimizer)

    rng, z_key, eval_rng, eval_rng2 = random.split(rng, 4)
    z = random.normal(z_key, (64, FLAGS.latents))

    # z0 = np.zeros(z.shape)
    # z0_vals = np.linspace(-1.5, 1.5, 8)
    # n_z0_vals = len(z0_vals)
    # for i1 in range(n_z0_vals):
    #    for i2 in range(n_z0_vals):
    #        z0[i1 * n_z0_vals + i2, 0] = z0_vals[i1]
    #        z0[i1 * n_z0_vals + i2, 1] = z0_vals[i2]
    # z0 = jnp.asarray(z0)
    # z = z0

    steps_per_epoch = 60000 // FLAGS.batch_size

    mode = 'train'
    # mode = 'output_z'
    # mode = 'sample_for_z_grid'
    if mode == 'train':
        logging.info('Start training')
        for epoch in range(FLAGS.num_epochs):
            for _ in range(steps_per_epoch):
                batch = next(train_ds)
                rng, key = random.split(rng)
                optimizer = train_step(optimizer, batch['image'], key)

            metrics, comparison, sample = eval(optimizer.target, test_ds, z, eval_rng, eval_rng2)
            save_image(comparison, f'samples/reconstruction_vaeprob_{epoch}.png', nrow=8)
            save_image(sample, f'samples/sample_vaeprob_{epoch}.png', nrow=8)

            logging.info('eval epoch: {}, bce: {:.4f}, kld: {:.4f}, bpd: {:.4f}'.format(
                epoch + 1, metrics['bce'], metrics['kld'], metrics['bpd_sample']
            ))
            print('eval epoch: {}, bce: {:.4f}, kld: {:.4f}, bpd: {:.4f}'.format(
                epoch + 1, metrics['bce'], metrics['kld'], metrics['bpd_sample']
            ), file=sys.stderr)
        # checkpoints.save_checkpoint(f'ckpt_mnist_vaeprob_{FLAGS.latents}', optimizer.target, step=1, keep=np.inf)

    elif mode == 'output_z':
        params = checkpoints.restore_checkpoint(f'ckpt_mnist_vae_{FLAGS.latents}', params, step=1)
        means, logvars, labels = [], [], []
        for _ in range(steps_per_epoch):
            batch = next(train_ds)
            rng, key = random.split(rng)
            _, mean, logvar = model().apply({'params': params}, batch['image'], key)
            means.append(mean)
            logvars.append(logvar)
            labels.append(batch['label'])
        means = np.concatenate(means, axis=0)
        logvars = np.concatenate(logvars, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.save('vae_z0', {'mean': means, 'logvar': logvars, 'label': labels})

    elif mode == 'sample_for_z_grid':
        params = checkpoints.restore_checkpoint(f'ckpt_mnist_vae_{FLAGS.latents}', params, step=1)
        x = np.load('ae_z_grid_transformed_2.npy', allow_pickle=True).item()

        z0 = x['z0']
        z0 = z0.reshape(-1, z0.shape[-1])
        zT = x['zT']
        zT = zT.reshape(-1, zT.shape[-1])

        img = eval(params, None, z0, None)
        print(img.shape, file=sys.stderr)
        save_image(img, f'samples/sample_grid_vae_z0.png', nrow=int(np.sqrt(z0.shape[0])))
        img = eval(params, None, zT, None)
        print(img.shape, file=sys.stderr)
        save_image(img, f'samples/sample_grid_vae_zT.png', nrow=int(np.sqrt(z0.shape[0])))

    else:
        raise Exception()

    print('Finished', file=sys.stderr)


if __name__ == '__main__':
    app.run(main)
