# Random Slice Contrastive Learning (RaSCL)

## Licensing
Copied and adapted from SimCLRv2 (TF2 implementation) (<https://github.com/google-research/simclr>)

## Pretraining

### Dataset specification

RaSCL pretraining expects a dataset with a 3D monochrome dataset structured with the slice dimension on the last axis (spec [image_size,image_size,slices], ie for an OCT scan block of 5 B-scans resized to 224x224, the dataset array spec would be [224,224,5]. Scans/images should be keyed in dataset as 'image', and labels as 'label'. As RaSCL was designed for smaller, specialized datasets, it expects a local dataset by default.

Results presented were also generated using 8-fold cross-validation using stratified data. To preserve stratification, it's recommended to split data into stratified groups during dataset preparation. K-fold training splits should be labeled f'{train_label}{fold}' eg {'train0','train1',...'train7'} to utilize RaSCL's k-fold cross-validation method.

### Running pretraining 

RaSCL may be run from the run.py directory:

```bash
run run.py --train_mode=pretrain --data_dir=./data/pretrain_dataset1 --model_dir=./models/pretrain_model1 \
--rascl_slice_max=2 --train_epochs=500 --kfold_groups=8
```

## Finetuning

### Dataset specification 

RaSCL finetuning expects a dataset with 2D monochrome images (spec [image_size,image_size,1], ie for OCT B-scans scaled to 224x224, the dataset array spec would be [224,224,1]. Scans/images should be keyed in dataset as 'image', and labels as 'label'. As RaSCL was designed for smaller, specialized datasets, it expects a local dataset by default. 

As per pretraining, finetuning k-fold splits should be labeled.

### Running finetuning

A previously pretrained RaSCL model can be finetuned:

```bash
run run.py --train_mode=finetune --data_dir=./data/dataset1 --model_dir=./models/finetune_model1 \
--train_epochs=500 --kfold_groups=8
```
