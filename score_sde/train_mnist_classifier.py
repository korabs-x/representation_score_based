#!/usr/bin/env python3

import jax
import jax.numpy as jnp  # JAX NumPy

from flax import linen as nn  # The Linen API
from flax import optim  # Optimizers
import flax

import numpy as np  # Ordinary NumPy
import tensorflow_datasets as tfds  # TFDS for MNIST
from flax.training import checkpoints


class CNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)  # There are 10 classes in MNIST
        x = nn.log_softmax(x)
        return x


def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))


def create_optimizer(params, learning_rate, beta):
    optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
    optimizer = optimizer_def.create(params)
    return optimizer


def get_initial_params(key):
    init_shape = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = CNN().init(key, init_shape)['params']
    return initial_params


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return metrics


def get_datasets():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    # Split into training/test sets
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    # Convert to floating-points
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.0 * 2. - 1.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.0 * 2. - 1.
    return train_ds, test_ds


@jax.jit
def train_step(optimizer, batch):
    def loss_fn(params):
        logits = CNN().apply({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = compute_metrics(logits, batch['label'])
    return optimizer, metrics


# JIT compile
@jax.jit
def eval_step(params, batch):
    logits = CNN().apply({'params': params}, batch['image'])
    return compute_metrics(logits, batch['label'])


def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    batch_metrics = []

    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        optimizer, metrics = train_step(optimizer, batch)
        batch_metrics.append(metrics)

    training_batch_metrics = jax.device_get(batch_metrics)
    training_epoch_metrics = {
        k: np.mean([metrics[k] for metrics in training_batch_metrics])
        for k in training_batch_metrics[0]}

    print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

    return optimizer, training_epoch_metrics


def eval_model(model, test_ds):
    metrics = eval_step(model, test_ds)  # Evalue the model on the test set
    metrics = jax.device_get(metrics)
    eval_summary = jax.tree_map(lambda x: x.item(), metrics)
    return eval_summary['loss'], eval_summary['accuracy']


if __name__ == '__main__':
    train_ds, test_ds = get_datasets()

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    params = get_initial_params(init_rng)

    learning_rate = 0.1
    beta = 0.9
    num_epochs = 10
    batch_size = 32

    optimizer = create_optimizer(params, learning_rate=learning_rate, beta=beta)

    train = False
    if train:
        for epoch in range(1, num_epochs + 1):
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            optimizer, train_metrics = train_epoch(optimizer, train_ds, batch_size, epoch, input_rng)
            # Evaluate on the test set after each training epoch
            test_loss, test_accuracy = eval_model(optimizer.target, test_ds)
            print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))

        checkpoints.save_checkpoint('ckpt_mnist_classifier', optimizer.target, step=10, keep=np.inf)
    else:
        params = checkpoints.restore_checkpoint('ckpt_mnist_classifier', params, step=10)
        test_loss, test_accuracy = eval_model(params, test_ds)
        print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (10, test_loss, test_accuracy * 100))
