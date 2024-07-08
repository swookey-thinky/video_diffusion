# video_diffusion
Small video diffusion implementation using latest research papers. Built for testing on small datasets, but extensible to large datasets given enough GPU.

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

## Introduction

This repository contains implementations of different video diffusion models, starting with the seminal paper from Ho. et al. [Video Diffusion Models](https://arxiv.org/abs/2204.03458).

Due to the resource constraints of most models, we have decided to use the [Moving MNIST](https://www.cs.toronto.edu/~nitish/unsupervised_video/) dataset to train on. Moving MNIST is a simple dataset similar to MNIST, of digits which move around the screen. It is an unlabeled dataset, so we do not have access to text labels to determine which digits are moving around the screen, but we will address that deficiency as well. We train at a reduced resolution of `32x32`, due to the resource constraints that most models require. This allows us to train most diffusion models on a T4 instance, which is free to run on [Google Colab](https://colab.research.google.com/). We limit training and sample generation to 16 frames, even though the source dataset contains 20 frames.

## Requirements

This package built using PyTorch and written in Python 3. To setup an environment to run all of the lessons, we suggest using conda or venv:

```
> python3 -m venv video_diffusion_env
> source video_diffusion_env/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt
```

All lessons are designed to be run from the root of the repository, and you should set your python path to include the repository root:

```
> export PYTHONPATH=$(pwd)
```

If you have issues with PyTorch and different CUDA versions on your instance, make sure to install the correct version of PyTorch for the CUDA version on your machine. For example, if you have CUDA 11.8 installed, you can install PyTorch using:

```
> pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Results

Sample from the original dataset:

![Moving MNIST](https://drive.google.com/uc?export=view&id=1FS9lEd6DPFJ4Ka7hUgqk2BlsJ8FzOdPE)

| Date  | Name  | Paper | Config | Results | Details
| :---- | :---- | ----- | ------ | ----- | -----
| April 2022 | Video Diffusion Models | [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | [vdm.yaml](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/vdm.yaml) | ![Video Diffusion Models](https://drive.google.com/uc?export=view&id=1pF6WVY8_dlGudxZIsml3VWPxbUs0ONfa) | `100k steps @ batch size 8`
| September 2022 | Make-A-Video | [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792) |  |  | 
