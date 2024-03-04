# coding=utf-8
# Copyright 2020 The SimCLR Authors.
# Copyright 2023 The RaSCL Authors. 
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data pipeline."""

import functools
from absl import flags
from absl import logging

import data_util
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


def build_input_fn(builder, global_batch_size, topology, is_training, is_val=False, kfold = None):
    """Build input function.

    Args:
      builder: TFDS builder for specified dataset.
      global_batch_size: Global batch size.
      topology: An instance of `tf.tpu.experimental.Topology` or None.
      is_training: Whether to build in training mode.

    Returns:
      A function that accepts a dict of params and returns a tuple of images and
      features, to be used as the input_fn in TPUEstimator.
    """

    def _input_fn(input_context):
        """Inner input function."""
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        logging.info("Global batch size: %d", global_batch_size)
        logging.info("Per-replica batch size: %d", batch_size)
        preprocess_fn_pretrain = get_preprocess_fn(
            is_training, is_val, is_pretrain=True
        )
        preprocess_fn_finetune = get_preprocess_fn(
            is_training, is_val, is_pretrain=False
        )
        if FLAGS.finetune_regressor:
          num_classes = 1
        else:
          num_classes = builder.info.features["label"].num_classes

        def map_fn(image, label):
            """Produces multiple transformations of the same batch."""
            if (is_training or is_val) and FLAGS.train_mode == "pretrain":
                xs = []
                for _ in range(2):  # Two transformations
                    xs.append(preprocess_fn_pretrain(image))
                image = tf.concat(xs, -1)
            else:
                image = preprocess_fn_finetune(image)
            if FLAGS.finetune_regressor:
              label = tf.cast(label, dtype=tf.float32)
            else:
              label = tf.one_hot(label, num_classes)
              
            return image, label

        def map_fn_rascl(data):
            """Takes random slices within the same batch and transforms."""
            xs = []
            # Maximum channel index to prevent slice overflow.
            chtotal = data['image'].shape[-1]
            # Minimum channel index to prevent slice underflow.
            chmin = tf.cast([0],'int64')
            chmax = tf.cast([chtotal-1],'int64')
            # First selection randomly taken from all available slices.
            ch1 = tf.random.uniform(shape=[1], 
                                    minval=0,
                                    maxval=chtotal,
                                    dtype=tf.int64)
            # Getting delta channel value
            dch = tf.random.uniform(shape=[1], 
                                    minval=-FLAGS.rascl_slice_max,
                                    maxval=FLAGS.rascl_slice_max,
                                    dtype=tf.int64)
            # Sets second channel index with respect to possible range.
            ch2 = tf.reduce_max(
                tf.concat([tf.reduce_min(tf.concat([chmax ,ch1+dch],axis=-1),axis=-1)[...,tf.newaxis],
                                           chmin],axis=-1),axis=-1)[...,tf.newaxis]
            chs = [ch1,ch2]
            for i in range(2):  # Two transformations
                gathered = tf.gather(data['image'],chs[i],axis=-1)
                xs.append(preprocess_fn_pretrain(gathered))
            image = tf.concat(xs, -1)
            label = tf.one_hot(data['label'], num_classes)
            return image, label

        logging.info("num_input_pipelines: %d", input_context.num_input_pipelines)
        if FLAGS.kfold_groups == 0:
          dataset = builder.as_dataset(
              split=FLAGS.train_split if is_training else FLAGS.eval_split,
              shuffle_files=is_training,
              as_supervised=False if FLAGS.rascl_pretrain and FLAGS.train_mode == "pretrain" else True,
              # Passing the input_context to TFDS makes TFDS read different parts
              # of the dataset on different workers. We also adjust the interleave
              # parameters to achieve better performance.
              read_config=tfds.ReadConfig(
                  interleave_cycle_length=32,
                  interleave_block_length=1,
                  input_context=input_context,
              ),
              )
        else:
          k_groups = [x for x in range(FLAGS.kfold_groups)]
          print(kfold)
          k_groups.pop(kfold)
          if is_training:
            dataset = builder.as_dataset(split=[f'{FLAGS.train_split}{i}' for i in k_groups],
                                        shuffle_files = is_training,
                                        as_supervised = False if FLAGS.rascl_pretrain and FLAGS.train_mode == "pretrain" else True,
                                        read_config=tfds.ReadConfig(
                                            interleave_cycle_length=32,
                                            interleave_block_length=1,
                                            input_context=input_context,
                                        ),
                                        )
            dslist = tf.data.Dataset.from_tensor_slices(dataset)
            # 2. extract all elements from datasets and concat them into one dataset
            dataset = dslist.interleave(lambda x: x, cycle_length=1,
                                        num_parallel_calls=tf.data.AUTOTUNE)
          else: 
            dataset = builder.as_dataset(split=f'{FLAGS.train_split}{kfold}',
                                        shuffle_files = is_training,
                                        as_supervised = False if FLAGS.rascl_pretrain and FLAGS.train_mode == "pretrain" else True,
                                        read_config=tfds.ReadConfig(
                                            interleave_cycle_length=32,
                                            interleave_block_length=1,
                                            input_context=input_context,
                                        ),
                                        )
        if FLAGS.cache_dataset:
            dataset = dataset.cache()
        if is_training:
            options = tf.data.Options()
            options.deterministic = False
            options.experimental_slack = True
            dataset = dataset.with_options(options)
            buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
            dataset = dataset.shuffle(batch_size * buffer_multiplier)
            dataset = dataset.repeat(-1)
        if is_val:
            options = tf.data.Options()
            options.deterministic = False
            options.experimental_slack = True
            dataset = dataset.with_options(options)
            dataset = dataset.repeat(-1)
        if FLAGS.rascl_pretrain and FLAGS.train_mode == "pretrain":
            dataset = dataset.map(
                map_fn_rascl, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        else:
            dataset = dataset.map(
                map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        dataset = dataset.batch(batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    return _input_fn


def build_distributed_dataset(
    builder, batch_size, is_training, strategy, topology, is_val=False, kfold=None
):
    input_fn = build_input_fn(builder, batch_size, topology, is_training, is_val, kfold)
    return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, is_val, is_pretrain):
    """Get function that accepts an image and returns a preprocessed image."""
    # Disable test cropping for small images (e.g. CIFAR)
    if FLAGS.image_size <= 32:
        test_crop = False
    else:
        test_crop = True
    if is_val:
        is_training = True
    return functools.partial(
        data_util.preprocess_image,
        height=FLAGS.image_size,
        width=FLAGS.image_size,
        is_training=is_training,
        test_crop=test_crop
    )
