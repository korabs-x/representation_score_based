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
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import sys
from sklearn import datasets
import logging


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC)
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, additional_dim=None, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      additional_dim: An integer or `None`. If present, add one additional dimension to the output data,
        which equals the number of steps jitted together.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                         f'the number of devices ({jax.device_count()})')

    per_device_batch_size = batch_size // jax.device_count()
    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1
    # Create additional data dimension when jitting multiple steps together
    if additional_dim is None:
        batch_dims = [jax.local_device_count(), per_device_batch_size]
    else:
        batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]

    train_ds = None
    eval_ds = None
    # Create dataset builders for each dataset.
    if config.data.dataset == 'CIFAR10':
        dataset_builder = tfds.builder('cifar10')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    elif config.data.dataset == 'toydata':
        """
        num_ch = config.data.num_channels
        mean = (1., 1.)
        std = 0.1
        n_train_samples = 6000
        train_dist = np.random.normal(mean, std, size=(n_train_samples, num_ch))
        train_label = (np.random.uniform(0, 1, n_train_samples) < 0.1) * 1.
        train_np = train_dist * (train_label[:, None] * 2 - 1)  # train_label[:, None]

        n_test_samples = 10000 + 1
        test_dist = np.random.normal(mean, std, size=(n_test_samples, num_ch))
        test_label = (np.random.uniform(0, 1, n_test_samples) < 0.1) * 1.
        eval_np = test_dist * (test_label[:, None] * 2 - 1)

        train_ds = tf.data.Dataset.from_tensor_slices(
            {'image': train_np.reshape(n_train_samples, 1, 1, num_ch), 'label': train_label})
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {'image': eval_np.reshape(n_test_samples, 1, 1, num_ch), 'label': test_label})
        """
        n_train_samples = 10000
        n_test_samples = 10000 + 1

        toydata_version = 3

        num_ch = config.data.num_channels

        if toydata_version == 5:
            x = 1.
            y = 2.
            means = np.array(
                [(-3 * x, -y), (-3 * x, 0.), (-3 * x, y), (-x, -y), (-x, 0.), (-x, y), (x, -y), (x, 0.), (x, y),
                 (3 * x, -y), (3 * x, 0.), (3 * x, y)])
            std = np.array([0.05, 0.5])
            ps = np.array([1. for i in range(len(means))])
            ps /= ps.sum()

        elif toydata_version == 4:
            x = 1.
            y = 1.5
            means = np.array(
                [(-x, -y), (-x, 0.), (-x, y), (x, -y), (x, 0.), (x, y)])
            std = np.array([0.3, 0.5])

            ps = np.array([1. for i in range(len(means))])
            ps /= ps.sum()

            """
            std = .15
            n_train_samples = 1000
            # train_label = np.zeros(shape=n_train_samples)
            train_dist, train_label = datasets.make_moons(n_samples=n_train_samples, noise=std)
            train_np = train_dist

            n_test_samples = 10000 + 1
            # test_label = np.zeros(shape=n_test_samples)
            test_dist, test_label = datasets.make_moons(n_samples=n_test_samples, noise=std)
            eval_np = test_dist
            """
        else:
            if toydata_version == 2:
                means = np.array(
                    [(-1., -1.), (-1., 0.), (-1., 1.), (0., -1.), (0., 0.), (0., 1.), (1., -1.), (1., 0.), (1., 1.)])
                std = 0.2
            elif toydata_version == 3:
                sin60 = 0.866
                means = np.array(
                    [(1., 0.), (sin60, 0.5), (0.5, sin60), (0., 1.), (-0.5, sin60), (-sin60, 0.5), (-1., 0.),
                     (-sin60, -0.5), (-0.5, -sin60), (0., -1.), (0.5, -sin60), (sin60, -0.5)])
                std = 0.1
            ps = np.array([2. - (i % 2) for i in range(len(means))])
            ps /= ps.sum()

        train_label = np.random.choice(list(range(len(means))), p=ps, size=n_train_samples)
        train_dist = np.random.normal(means[train_label], std, size=(n_train_samples, num_ch))
        train_np = train_dist

        test_label = np.random.choice(list(range(len(means))), p=ps, size=n_test_samples)
        test_dist = np.random.normal(means[test_label], std, size=(n_test_samples, num_ch))
        eval_np = test_dist

        train_ds = tf.data.Dataset.from_tensor_slices(
            {'image': train_np.reshape(n_train_samples, 1, 1, num_ch), 'label': train_label})
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {'image': eval_np.reshape(n_test_samples, 1, 1, num_ch), 'label': test_label})

    elif config.data.dataset == 'toydatav1':
        num_ch = config.data.num_channels
        mean = (1., 1.)
        std = 0.1
        n_train_samples = 6000
        train_dist = np.random.normal(mean, std, size=(n_train_samples, num_ch))
        train_label = (np.random.uniform(0, 1, n_train_samples) < 0.2) * 1.
        train_np = train_dist * (train_label[:, None] * 2 - 1)  # train_label[:, None]

        n_test_samples = 10000 + 1
        test_dist = np.random.normal(mean, std, size=(n_test_samples, num_ch))
        test_label = (np.random.uniform(0, 1, n_test_samples) < 0.2) * 1.
        eval_np = test_dist * (test_label[:, None] * 2 - 1)

        train_ds = tf.data.Dataset.from_tensor_slices(
            {'image': train_np.reshape(n_train_samples, 1, 1, num_ch), 'label': train_label})
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {'image': eval_np.reshape(n_test_samples, 1, 1, num_ch), 'label': test_label})

    elif config.data.dataset == 'aedata':
        num_ch = config.data.num_channels

        x = np.load(f'ae_z0.npy', allow_pickle=True).item()

        data = x['z']
        print(("data.shape", data.shape), file=sys.stderr)

        n_train_samples = int(data.shape[0] * 0.8)
        n_test_samples = data.shape[0] - n_train_samples

        train_label = x['label'][:n_train_samples]
        train_dist = data[:n_train_samples]
        train_np = train_dist

        test_label = x['label'][-n_test_samples:]
        test_dist = data[-n_test_samples:]
        eval_np = test_dist

        train_ds = tf.data.Dataset.from_tensor_slices(
            {'image': train_np.reshape(n_train_samples, 1, 1, num_ch), 'label': train_label})
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {'image': eval_np.reshape(n_test_samples, 1, 1, num_ch), 'label': test_label})

    elif config.data.dataset == 'aedata_test':
        num_ch = config.data.num_channels

        x = np.load(f'ae_z0.npy', allow_pickle=True).item()

        data = x['z']
        print(("data.shape", data.shape), file=sys.stderr)

        n_train_samples = data.shape[0]
        n_test_samples = data.shape[0]

        train_label = x['label'][:]
        train_dist = data[:]
        train_np = train_dist

        test_label = x['label'][:]
        test_dist = data[:]
        eval_np = test_dist

        train_ds = tf.data.Dataset.from_tensor_slices(
            {'image': train_np.reshape(n_train_samples, 1, 1, num_ch), 'label': train_label})
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {'image': eval_np.reshape(n_test_samples, 1, 1, num_ch), 'label': test_label})

    elif config.data.dataset == 'vaedata':
        num_ch = config.data.num_channels

        x = np.load(f'vae_z0.npy', allow_pickle=True).item()

        logvar = x['logvar']
        mean = x['mean']
        std = np.exp(0.5 * logvar)
        eps = np.random.normal(size=logvar.shape)
        data = mean + eps * std
        print(("data.shape", data.shape), file=sys.stderr)

        n_train_samples = int(data.shape[0] * 0.8)
        n_test_samples = data.shape[0] - n_train_samples

        train_label = x['label'][:n_train_samples]
        train_dist = data[:n_train_samples]
        train_np = train_dist

        test_label = x['label'][-n_test_samples:]
        test_dist = data[-n_test_samples:]
        eval_np = test_dist

        train_ds = tf.data.Dataset.from_tensor_slices(
            {'image': train_np.reshape(n_train_samples, 1, 1, num_ch), 'label': train_label})
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {'image': eval_np.reshape(n_test_samples, 1, 1, num_ch), 'label': test_label})

    elif config.data.dataset == 'MNIST':
        dataset_builder = tfds.builder('mnist')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            pad_images = True
            if pad_images:
                pad_w = (config.data.image_size - 28) // 2
                img = tf.pad(img, tf.constant([[pad_w, pad_w], [pad_w, pad_w], [0, 0]]))
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    elif config.data.dataset == 'binMNIST':
        dataset_builder = tfds.builder('binarized_mnist')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.cast(img, tf.float32)
            # img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    elif config.data.dataset == 'SVHN':
        dataset_builder = tfds.builder('svhn_cropped')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)


    elif config.data.dataset == 'CELEBA':
        dataset_builder = tfds.builder('celeb_a')
        train_split_name = 'train'
        eval_split_name = 'validation'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = central_crop(img, 140)
            img = resize_small(img, config.data.image_size)
            return img

    elif config.data.dataset == 'LSUN':
        dataset_builder = tfds.builder(f'lsun/{config.data.category}')
        train_split_name = 'train'
        eval_split_name = 'validation'

        if config.data.image_size == 128:
            def resize_op(img):
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = resize_small(img, config.data.image_size)
                img = central_crop(img, config.data.image_size)
                return img

        else:
            def resize_op(img):
                img = crop_resize(img, config.data.image_size)
                img = tf.image.convert_image_dtype(img, tf.float32)
                return img

    elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
        dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
        train_split_name = eval_split_name = 'train'

    else:
        raise NotImplementedError(
            f'Dataset {config.data.dataset} not yet supported.')

    # Customize preprocess functions for each dataset.
    if config.data.dataset in ['FFHQ', 'CelebAHQ']:
        def preprocess_fn(d):
            sample = tf.io.parse_single_example(d, features={
                'shape': tf.io.FixedLenFeature([3], tf.int64),
                'data': tf.io.FixedLenFeature([], tf.string)})
            data = tf.io.decode_raw(sample['data'], tf.uint8)
            data = tf.reshape(data, sample['shape'])
            data = tf.transpose(data, (1, 2, 0))
            img = tf.image.convert_image_dtype(data, tf.float32)
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
            return dict(image=img, label=None)

    elif config.data.dataset in ['toydata', 'toydatav1', 'aedata', 'vaedata', 'aedata_test']:
        def preprocess_fn(d):
            return d
    else:
        def preprocess_fn(d):
            """Basic preprocessing function scales data to [0, 1) and randomly flips."""
            img = resize_op(d['image'])
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

            return dict(image=img, label=d.get('label', img[0, 0, :] * 0.))  # , id=d['id'])

    def create_dataset(dataset_builder, split, ds=None):
        if ds is None:
            dataset_options = tf.data.Options()
            dataset_options.experimental_optimization.map_parallelization = True
            dataset_options.experimental_threading.private_threadpool_size = 48
            dataset_options.experimental_threading.max_intra_op_parallelism = 1
            read_config = tfds.ReadConfig(options=dataset_options)
            if isinstance(dataset_builder, tfds.core.DatasetBuilder):
                dataset_builder.download_and_prepare()
                ds = dataset_builder.as_dataset(
                    split=split, shuffle_files=True, read_config=read_config)
            else:
                ds = dataset_builder.with_options(dataset_options)

        allowed_labels = getattr(config.data, 'allowed_labels', None)
        n_labelled_samples = getattr(config.data, 'n_labelled_samples', -1)
        is_eval_split = split == 'test'
        if n_labelled_samples >= 0 and not is_eval_split:
            n_classes = 10
            n_per_class = np.zeros(n_classes)
            n_allowed_per_class = n_labelled_samples // n_classes
            label_ids = []

            for d in ds:
                label = d['label'].numpy()
                if n_per_class[label] < n_allowed_per_class:
                    n_per_class[label] += 1
                    label_ids.append(d['id'].numpy())
                if n_per_class.min() == n_allowed_per_class:
                    break

            logging.info(("ids of labelled samples:", label_ids))
            for _ in range(30):
                logging.info(("len(ids) of labelled samples:", len(label_ids)))

            label_ids = tf.convert_to_tensor(label_ids)

            def preprocess_label_fn(d):
                label = d['label']
                if not tf.math.reduce_any(d['id'] == label_ids):
                    label = tf.cast(-1, tf.dtypes.int64)
                """
                new_n_per_class = n_per_class + tf.cast((label == class_tensor), tf.dtypes.int64)
                if tf.math.reduce_sum(tf.cast(new_n_per_class > n_allowed_per_class, tf.dtypes.int32)) > 0:
                    # remove label
                    label = tf.cast(-1, tf.dtypes.int64)
                else:
                    n_per_class = new_n_per_class
                """
                return dict(image=d['image'], label=label)  # , id=d['id'])

            ds = ds.map(preprocess_label_fn)
            if getattr(config.data, 'only_include_labeled_samples', False) or getattr(config.training, 'train_clf', False):
                allowed_labels = list(range(n_classes))

        if allowed_labels is not None:
            allowed_labels = tf.cast(tf.constant(allowed_labels), tf.dtypes.int64)

            def predicate(x):
                label = x['label']
                isallowed = tf.equal(allowed_labels, label)
                reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
                return tf.greater(reduced, tf.constant(0.))

            ds = ds.filter(predicate)

        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for batch_size in reversed(batch_dims):
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

    if train_ds is None:
        train_ds = create_dataset(dataset_builder, train_split_name)
        eval_ds = create_dataset(dataset_builder, eval_split_name)
    else:
        train_ds = create_dataset(None, None, ds=train_ds)
        eval_ds = create_dataset(None, None, ds=eval_ds)
        dataset_builder = None

    return train_ds, eval_ds, dataset_builder
