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
"""Training utilities."""

from absl import logging,flags

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

def update_pretrain_metrics_train(contrast_loss, contrast_acc, contrast_entropy,
                                  loss, logits_con, labels_con):
  """Updated pretraining metrics."""
  contrast_loss.update_state(loss)

  contrast_acc_value = tf.equal(
      tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
  contrast_acc_value = tf.reduce_mean(tf.cast(contrast_acc_value, tf.float32))
  contrast_acc.update_state(contrast_acc_value)

  prob_con = tf.nn.softmax(logits_con)
  entropy_con = -tf.reduce_mean(
      tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))
  contrast_entropy.update_state(entropy_con)

def update_pretrain_metrics_val(contrast_loss_val, contrast_acc_val, contrast_entropy_val,
                                  loss, logits_con, labels_con):
  """Updated pretraining metrics."""
  contrast_loss_val.update_state(loss)

  contrast_acc_value = tf.equal(
      tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
  contrast_acc_value = tf.reduce_mean(tf.cast(contrast_acc_value, tf.float32))
  contrast_acc_val.update_state(contrast_acc_value)

  prob_con = tf.nn.softmax(logits_con)
  entropy_con = -tf.reduce_mean(
      tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))
  contrast_entropy_val.update_state(entropy_con)  

def update_pretrain_metrics_eval(contrast_loss_metric,
                                 contrastive_top_1_accuracy_metric,
                                 contrastive_top_5_accuracy_metric,
                                 contrast_loss, logits_con, labels_con):
  contrast_loss_metric.update_state(contrast_loss)
  contrastive_top_1_accuracy_metric.update_state(
      tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
  contrastive_top_5_accuracy_metric.update_state(labels_con, logits_con)


def update_finetune_metrics_train(supervised_loss_metric, supervised_acc_metric,
                                  loss, labels, logits):
  supervised_loss_metric.update_state(loss)
  
  if FLAGS.finetune_regressor:
    label_acc = loss
  else:
    label_acc = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, axis=1))
    label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
  supervised_acc_metric.update_state(label_acc)
  
def update_finetune_metrics_val(supervised_loss_metric_val, supervised_acc_metric_val,
                                  loss, labels, logits):
  supervised_loss_metric_val.update_state(loss)
  if FLAGS.finetune_regressor:
    label_acc = loss
  else:
    label_acc = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, axis=1))
    label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
  supervised_acc_metric_val.update_state(label_acc)


def update_finetune_metrics_eval(label_accuracy_metrics, outputs, labels):
  label_accuracy_metrics.update_state(
      tf.argmax(labels, 1), tf.argmax(outputs, axis=1))


def _float_metric_value(metric):
  """Gets the value of a float-value keras metric."""
  return metric.result().numpy().astype(float)


def log_and_write_metrics_to_summary(all_metrics, global_step):
  for metric in all_metrics:
    metric_value = _float_metric_value(metric)
    logging.info('Step: [%d] %s = %f', global_step, metric.name, metric_value)
    tf.summary.scalar(metric.name, metric_value, step=global_step)
