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

| Date  | Name  | Paper | Config | Results | Instructions
| :---- | :---- | ----- | ------ | ----- | -----
| April 2022 | Video Diffusion Models | [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | [config](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/video_diffusion_models.yaml) | ![Video Diffusion Models](https://drive.google.com/uc?export=view&id=1pF6WVY8_dlGudxZIsml3VWPxbUs0ONfa) | [instructions](https://github.com/swookey-thinky/video_diffusion/blob/main/docs/video_diffusion_models.md)
| May 2022 | CogVideo | [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868) | | |
| September 2022 | Make-A-Video | [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792) | [config](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/make_a_video.yaml) |  | 
| October 2022 | Imagen Video | [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303) | | |
| October 2022 | Phenaki | [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/abs/2210.02399) | | |
| December 2022 | Tune-A-Video  | [Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2212.11565) | | |
| March 2023 | Text2Video-Zero | [Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://arxiv.org/abs/2303.13439) | | |
| April 2023 | Video LDM | [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818) | | |
| July 2023 | AnimateDiff | [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725) | | |
| August 2023 | ModelScopeT2V | [ModelScope Text-to-Video Technical Report](https://arxiv.org/abs/2308.06571) | | |
| September 2023 | Show-1 | [Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2309.15818) | | |
| October 2023 | VideoCrafter 1 | [VideoCrafter1: Open Diffusion Models for High-Quality Video Generation](https://arxiv.org/abs/2310.19512) | | |
| November 2023 | Emu Video | [Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning](https://arxiv.org/abs/2311.10709) | | |
| November 2023 | | [Decouple Content and Motion for Conditional Image-to-Video Generation](https://arxiv.org/abs/2311.14294) | | |
| November 2023 | Stable Video Diffusion | [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127) | | |
| December 2023 | VideoBooth | [VideoBooth: Diffusion-based Video Generation with Image Prompts](https://arxiv.org/abs/2312.00777) | | |
| December 2023 | LivePhoto | [LivePhoto: Real Image Animation with Text-guided Motion Control](https://arxiv.org/abs/2312.02928) | | |
| December 2023 | HiGen | [Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation](https://arxiv.org/abs/2312.04483) | | |
| December 2023 | AnimateZero | [AnimateZero: Video Diffusion Models are Zero-Shot Image Animators](https://arxiv.org/abs/2312.03793) | | |
| December 2023 | W.A.L.T | [Photorealistic Video Generation with Diffusion Models](https://arxiv.org/abs/2312.06662) | | |
| December 2023 | VideoLCM | [VideoLCM: Video Latent Consistency Model](https://arxiv.org/abs/2312.09109) | | |
| January 2024 | Latte | [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048) | | |
| January 2024 | VideoCrafter 2 | [VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models](https://arxiv.org/abs/2401.09047) | | |
| January 2024 | Lumiere | [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945) | | |
| February 2024 | Video-LaVIT | [Video-LaVIT: Unified Video-Language Pre-training with Decoupled Visual-Motional Tokenization](https://arxiv.org/abs/2402.03161) | | |
| February 2024 | Snap Video | [Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis](https://arxiv.org/abs/2402.14797) | | |
| April 2024 | TI2V-Zero | [TI2V-Zero: Zero-Shot Image Conditioning for Text-to-Video Diffusion Models](https://arxiv.org/abs/2404.16306) | | |
| May 2024 | Vidu | [Vidu: a Highly Consistent, Dynamic and Skilled Text-to-Video Generator with Diffusion Models](https://arxiv.org/abs/2405.04233) | | |
| May 2024 | FIFO-Diffusion | [FIFO-Diffusion: Generating Infinite Videos from Text without Training](https://arxiv.org/abs/2405.11473) | | |
