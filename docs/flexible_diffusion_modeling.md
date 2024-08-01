# Flexible Diffusion Modeling of Long Videos

In this example we introduce Flexible Diffusion Modeling, from the paper [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495).

## Introduction

Flexible diffusion modeling introduces a novel training scheme in order to conditionally sample arbitrarily long videos. For example, from a training set of only 16 frames per video, we can generate sampled videos as long as we want! They achieve this by randomly sampling frames to condition on in the training data set, and use those frames to generate the "missing" frames. The paper calls these "observations" (the frames we condition on) and "latents" (the frames that are generated conditioned on the observations).

The authors used a space-time factorized UNet, similar to [Video Diffusion Models](https://arxiv.org/abs/2204.03458), but preferred the epsilon paramterization and discrete noise scheduling, so we follow that here.

## Configuration File

The configuration file is located in [Flexible Diffusion Models](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/flexible_diffusion_models.yaml).

## Training

To train the video diffusion model, use:

```
> python training/moving_mnist/train.py --config_path configs/moving_mnist/flexible_diffusion_models.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 8.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python training/moving_mnist/sample.py --config_path configs/moving_mnist/flexible_diffusion_models.yaml --num_samples 8 --checkpoint output/moving_mnist/flexible_diffusion_models/diffusion-100000.pt
```

Output will be saved to the `output/moving_mnist/sample/flexible_diffusion_models` directory.

However, you can also autoregressively sample arbitrary length videos, using a defined sampling scheme. For example, in `configs/sampling_schemes/autoregressive.yaml`, we have defined an autoregressive sampling scheme that generates a 160 frame video, conditioning each segment on the previous 8 frames in the video. To run this, run:

```
> python training/moving_mnist/sample.py --config_path configs/moving_mnist/flexible_diffusion_models.yaml --num_samples 16 --checkpoint output/moving_mnist/flexible_diffusion_models/diffusion-100000.pt --sampling_scheme_path configs/sampling_schemes/autoregressive.yaml
```

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/flexible_diffusion_models.yaml) | [google drive](https://drive.google.com/file/d/1rDX-sioy4B3uUFjQfQnmZ5ASzIE7V5gb/view?usp=sharing) | ![Flexible Diffusion Models](https://drive.google.com/uc?export=view&id=1B2raR3_suRf8qAUP4jzi8YIwka-UHwrU)

## Other Resources

The authors released their original source code [here](https://github.com/plai-group/flexible-video-diffusion-modeling).
