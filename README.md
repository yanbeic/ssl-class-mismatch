# ssl-class-mismatch

This repository contains the code for 
[Semi-Supervised Learning under Class Distribution Mismatch](https://yanbeic.github.io/Doc/AAAI20-ChenY.pdf), which is built upon the implementation from [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170).

The code is designed to run on Python 3 using the dependencies listed in `requirements.txt`.
You can install the dependencies by running `pip3 install -r requirements.txt`.

# Prepare datasets

For SVHN and CIFAR-10, we provide scripts to automatically download and preprocess the data.
We also provide a script to create "label maps", which specify which entries of the dataset should be treated as labeled and unlabeled. Both of these scripts use an explicitly chosen random seed, so the same dataset order and label maps will be created each time. The random seeds can be overridden, for example to test robustness to different labeled splits.
Run those scripts as follows:

```sh
python3 build_tfrecords.py --dataset_name=cifar10
python3 build_label_map.py --dataset_name=cifar10
python3 build_tfrecords.py --dataset_name=svhn
python3 build_label_map.py --dataset_name=svhn
```

For ImageNet 32x32 (only used in the fine-tuning experiment), you'll first need to download the 32x32 version of the ImageNet dataset by following the instructions [here](https://patrykchrabaszcz.github.io/Imagenet32/).
Unzip the resulting files and put them in a directory called 'data/imagenet_32'.
You'll then need to convert those files (which are pickle files) into .npy files.
You can do this by executing:

```sh
mkdir data/imagenet_32
unzip Imagenet32_train.zip -d data/imagenet_32
unzip Imagenet32_val.zip -d data/imagenet_32
python3 convert_imagenet.py
```

Then you can build the TFRecord files like so:

```sh
python3 build_tfrecords.py --dataset_name=imagenet_32
```

ImageNet32x32 is the only dataset which must be downloaded manually, due to licensing issues.

# Running experiments

All of the experiments in our paper are accompanied by a .yml file in `runs/`.These .yml files are intended to be used with [tmuxp](https://github.com/tmux-python/tmuxp), which is a session manager for tmux.
They essentially provide a simple way to create a tmux session with all of the relevant tasks running (model training and evaluation).

For example, for the UASD model in [Semi-Supervised Learning under Class Distribution Mismatch](https://yanbeic.github.io/Doc/AAAI20-ChenY.pdf), you could run 

```sh
tmuxp load run-uasd/cifar10-4000.yml
```

If you want to run an experiment evaluating VAT with 500 labels as shown in Figure 3, you could run

```sh
tmuxp load runs/figure-3-svhn-500-vat.yml
```

Of course, you can also run the code without using tmuxp.
Each .yml file specifies the commands needed for running each experiment.
For example, the file listed above `runs/figure-3-svhn-500-vat.yml` runs

```sh
CUDA_VISIBLE_DEVICES=0 python3 train_model.py --verbosity=0 --primary_dataset_name='svhn' --secondary_dataset_name='svhn' --root_dir=/mnt/experiment-logs/figure-3-svhn-500-vat --n_labeled=500 --consistency_model=vat --hparam_string=""  2>&1 | tee /mnt/experiment-logs/figure-3-svhn-500-vat_train.log
CUDA_VISIBLE_DEVICES=1 python3 evaluate_model.py --split=test --verbosity=0 --primary_dataset_name='svhn' --root_dir=/mnt/experiment-logs/figure-3-svhn-500-vat --consistency_model=vat --hparam_string=""  2>&1 | tee /mnt/experiment-logs/figure-3-svhn-500-vat_eval_test.log
CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py --split=valid --verbosity=0 --primary_dataset_name='svhn' --root_dir=/mnt/experiment-logs/figure-3-svhn-500-vat --consistency_model=vat --hparam_string=""  2>&1 | tee /mnt/experiment-logs/figure-3-svhn-500-vat_eval_valid.log
CUDA_VISIBLE_DEVICES=3 python3 evaluate_model.py --split=train --verbosity=0 --primary_dataset_name='svhn' --root_dir=/mnt/experiment-logs/figure-3-svhn-500-vat --consistency_model=vat --hparam_string=""  2>&1 | tee /mnt/experiment-logs/figure-3-svhn-500-vat_eval_train.log
```

Note that these commands are formulated to write out results to `/mnt/experiment-logs`.
