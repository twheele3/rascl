# coding=utf-8
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
import functools
import tensorflow.compat.v2 as tf
import data_util
import tensorflow_datasets as tfds
from tensorflow.python.summary.summary_iterator import summary_iterator
from sklearn.metrics import RocCurveDisplay, auc, r2_score, roc_curve
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import shutil

def norm(arr):
    return arr / arr.max()

def load_best_kfold(model_path, load_num = 0, keep_best = 1):
  """
  Reads tensorboard logs and determines best model fit based on accuracy divided by loss.
  Expects a kfold fit.
  load_num (int) : Integer or list of integers (start 0) for which best model to use.
  keep_best (int): If not None, keeps best n models and discards the rest.
  """
  ln = load_num
  load_step = []
  kfolds = max([int(i[3:]) for i in os.listdir(model_path) if 'val' in i]) + 1
  for kfold in range(kfolds):
    if type(load_num) is list:
      ln = load_num[kfold]
    path = os.path.join(model_path,f'val{kfold}/')
    reader = summary_iterator(os.path.join(path,os.listdir(path)[0]))
    values = {}
    for event in reader:
      step = event.step
      for val in event.summary.value:
        if val.tag not in values.keys():
          values[val.tag] = {}
        values[val.tag][step] = tf.make_ndarray(val.tensor)
    x = np.asarray([i for i in values['supervised_loss'].items()])
    y = np.asarray([i for i in values['supervised_acc'].items()])
    best_fits = np.flip(x[(y[:,1]/x[:,1]).argsort(),0].astype(np.int16))
    load_step.append(best_fits[ln])
    if keep_best is not None:
      for i in best_fits[keep_best:]:
          active_dir = os.path.join(model_path,f'kfold{kfold}')
          to_remove = os.path.join(active_dir, f'saved_model-{i}')
          if os.path.exists(to_remove):
            shutil.rmtree(os.path.join(active_dir, f'saved_model-{i}'))
  return load_step
  
def get_preprocess_fn(size):
    """Get function that accepts an image and returns a preprocessed image."""
    return functools.partial(
        data_util.preprocess_image,
        height=size,
        width=size,
        is_training=False,
        test_crop=False
    )
  
def load_dataset(data_dir, split = 'test'):
  """Loads images and labels from dataset located in data_dir, using designated split"""
  builder = tfds.builder_from_directory(builder_dir=data_dir)
  ds = builder.as_dataset(
      split=split, batch_size=builder.info.splits[split].num_examples
  )
  iterator = iter(ds)
  take = next(iterator)
  images = tf.cast(take["image"], dtype=tf.float32) / 255.0
  labels = take["label"]
  return images,labels,take
  
def run_best_kfold(model_path,images,labels):
  kfolds = max([int(i[3:]) for i in os.listdir(model_path) if 'val' in i]) + 1
  load_step = load_best_kfold(model_path)
  y_predicted = []
  for kfold in range(kfolds):
    model_path_full = os.path.join(
        model_path, f"kfold{kfold}/saved_model-{load_step[kfold]}"
    )

    model = tf.keras.models.load_model(model_path_full)
    features = images
    for layer in model.layers:
      features = layer(features)
    y_predicted.append(tf.keras.activations.softmax(features, axis=-1))
  return y_predicted

def plot_kfold_ensemble(y_predicted, labels, save_path, prefix, class_labels = None):
  x = labels
  kfolds = len(y_predicted)
  classes = y_predicted[0].shape[-1]
  if class_labels == None:
    class_labels = [i for i in range(y_predicted[0].shape[-1])]
  for c in range(classes):
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor("white")
    fpr, tpr, _ = roc_curve(y_true=x, 
                            y_score=np.asarray(y_predicted)[...,c].mean(axis=0), 
                            pos_label=c)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        label=f"consensus area = {roc_auc:0.3f}", lw=5
    )
    for i in range(kfolds):
        fpr, tpr, _ = roc_curve(y_true=x, 
                                y_score=y_predicted[i][...,c], 
                                pos_label=c)
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"kfold{i} area = {roc_auc:0.3f}", ls=':',lw=3, alpha = 0.8
        )
    plt.plot([0, 1], [0, 1], color="navy", ls="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC of class {class_labels[c]} of kfold ensemble for test data")
    plt.legend(loc="lower right", markerfirst=False)
    plt.savefig(f'{save_path}/{prefix} kfold ensemble class{c}.png', bbox_inches='tight')
    
def plot_kfold_summary(y_predicted, labels, save_path, prefix, class_labels = None):
  x = labels
  if class_labels == None:
    class_labels = [i for i in range(y_predicted[0].shape[-1])]
  classes = len(class_labels)
  y = np.asarray(y_predicted)[...,1].mean(axis=0)
  curlabel = ['Worse','Better']
  fig = plt.figure(figsize=(15, 5))
  fig.patch.set_facecolor("white")
  plt.subplot(1,3,1)
  fpr, tpr, _ = roc_curve(y_true=x, y_score=y, pos_label=1)
  roc_auc = auc(fpr, tpr)
  plt.plot(
      fpr,
      tpr,
      label=f"area under curve = {roc_auc:0.3f}", lw=5
  )
  plt.plot([0, 1], [0, 1], color="navy", ls="--")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver operating characteristic")
  plt.legend(loc="lower right", markerfirst=False)
  ax = fig.add_subplot(1,3,2)
  correct = (y > 0.5).astype(np.int16)
  vals = np.unique(labels.numpy())
  mtx = np.zeros((len(vals),len(vals)),dtype=np.int16)
  for i in vals:
      for j in vals:
          mtx[i,j] = sum(correct[labels.numpy()==i]==j)

  fmtx = mtx / mtx.sum(axis=1)[...,np.newaxis]
  cscaling = 0.75
  ax.matshow(np.ma.masked_where(np.eye(classes,dtype=bool),fmtx), 
             cmap=plt.cm.Oranges,vmin=0,vmax=1/cscaling)
  ax.matshow(np.ma.masked_where(~np.eye(classes,dtype=bool),fmtx), 
             cmap=plt.cm.Blues,vmin=0,vmax=1/cscaling)

  for i in range(classes):
      for j in range(classes):
          c = mtx[j,i]
          ax.text(i, j, str(c), va='center', ha='center',size=16, 
                  color = 'w' if fmtx[j,i]>(0.8/cscaling) else 'black')
  plt.ylabel('Actual diagnosis')
  plt.xlabel('Predicted diagnosis')
  ax.set_xticks(np.arange(classes))
  ax.set_xticklabels(class_labels)
  ax.set_yticks(np.arange(classes))
  ax.set_yticklabels(class_labels)
  ax.xaxis.set_ticks_position("bottom")
  plt.yticks(rotation=90)
  plt.title(f'Confusion matrix (n = {len(x)})')
  ax2 = fig.add_subplot(1,3,3)
  for i in range(classes):
      sns.kdeplot(y[x==i], linewidth=3,label=class_labels[i], fill=True, bw_adjust=0.3)
  plt.legend()
  for i in range(classes):
      sns.rugplot(y[x==i], height=0.1, linewidth=2, alpha=0.5, label=class_labels[i])
  ax2.set_xticks(np.arange(0,1.01,0.25))
  plt.xlabel('Softmax score for FTMH+ diagnosis')
  plt.ylabel('Density')
  plt.title('Score distribution')

  plt.savefig(f'{save_path}/{prefix} performance.png', bbox_inches='tight')
  
def pretrain_get_best(model_dir):
  # Reads event log data for validation to determine best step
  path = os.path.join(model_dir,'val')
  reader = summary_iterator(os.path.join(path,os.listdir(path)[0]))
  values = {}
  for event in reader:
      step = event.step
      for val in event.summary.value:
          if val.tag not in values.keys():
              values[val.tag] = {}
          values[val.tag][step] = tf.make_ndarray(val.tensor)
  fig = plt.figure(figsize=(5, 5))
  x = np.asarray([i for i in values['contrast_loss'].items()])
  plt.plot(x[:,0],norm(x[:,1]),label='Norm Loss')
  y = np.asarray([i for i in values['contrast_acc'].items()])
  plt.plot(y[:,0],y[:,1],label='Acc')
  plt.plot(y[:,0],norm(y[:,1]/(x[:,1]*np.log(y[:,0]))),label='Norm Acc / (Loss * Log(Epoch))')
  plt.xlabel('Epoch')
  plt.ylabel('Norm value')
  plt.legend()
  model_name = [i for i in model_dir.split('/') if len(i)>0][-1]
  # plt.savefig(os.path.join(model_dir, f'{model_name} pretrain val.png'), bbox_inches='tight')
  # Removes all but best checkpoint
  ckpt_best = int(y[:,0][norm(y[:,1]/(x[:,1]*np.log(y[:,0]))).argmax()])
  return ckpt_best

def plot_finetune_val(model_dir):
  kfolds = max([int(i[3:]) for i in os.listdir(model_dir) if 'val' in i]) + 1
  prefix = [i for i in model_dir.split('/') if len(i)>0][-1]
  fig = plt.figure(figsize=(8,(kfolds//2) * 4))
  for kfold in range(kfolds):
    path = os.path.join(model_dir,f'val{kfold}/')
    reader = summary_iterator(os.path.join(path,os.listdir(path)[0]))
    values = {}
    for event in reader:
      step = event.step
      for val in event.summary.value:
        if val.tag not in values.keys():
            values[val.tag] = {}
        values[val.tag][step] = tf.make_ndarray(val.tensor)
    plt.subplot(kfolds//2,2,kfold+1)
    x = np.asarray([i for i in values['supervised_loss'].items()])
    plt.plot(x[:,0],norm(x[:,1]),label='Norm SupervLoss')
    y = np.asarray([i for i in values['supervised_acc'].items()])
    plt.plot(y[:,0],y[:,1],label='Supervised Acc')
    plt.plot(y[:,0],norm(y[:,1]/x[:,1]),label='Norm Acc / Loss')
    if kfold == 1:
      plt.legend()
  plt.savefig(f'{model_dir}/{prefix} kfold training curve.png', bbox_inches = 'tight')
  