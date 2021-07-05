#!/usr/bin/env python3

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

"""Training and evaluation"""

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval", "both"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    print("main")
    # tf.config.experimental.set_visible_devices([], "GPU")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    if hasattr(FLAGS.config.model, 'single_t100') and FLAGS.config.model.single_t100 != "notset":
        FLAGS.config.model.single_t = float(FLAGS.config.model.single_t100) * 0.01
    if hasattr(FLAGS.config.model, 'lambda_z_sh') and FLAGS.config.model.lambda_z_sh != "notset":
        FLAGS.config.model.lambda_z = float(FLAGS.config.model.lambda_z_sh)
    if hasattr(FLAGS.config.data, 'allow_all_labels') and FLAGS.config.data.allow_all_labels:
        print("allow_all_labels is True")
        FLAGS.config.data.allowed_labels = None

    if FLAGS.mode in ["train", 'both']:
        # Create the working directory
        tf.io.gfile.makedirs(FLAGS.workdir)
        # Set logger so that it outputs to both console and file
        handler1 = logging.StreamHandler()
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = tf.io.gfile.GFile(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
        handler2 = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel('INFO')
        # Run the training pipeline
        run_lib.train(FLAGS.config, FLAGS.workdir)
        if FLAGS.mode == 'both':
            run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    print("start...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    app.run(main)
