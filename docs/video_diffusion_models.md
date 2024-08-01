# Video Diffusion Models

In this example we introduce Video Diffusion Models from the seminal paper [Video Diffusion Models](https://arxiv.org/abs/2204.03458).

## Introduction

Video Diffusion Models is a seminal paper from 2022 that adapts image diffusion models from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) into their equivalent space-time video diffusion formulation. Specifically, this paper introduces the concept of generating video frames in one shot, by expanding the image input with shape `[B, C, H, W]` to video input with shape `[B, C, F, H, W]`.

In order to accomplish this, the authors factorized the spatial 2D UNet in the score network into a space a time 3D Unet. The architectural changes to support a space-time factorized 3D UNet include:

-   Replacing all of the 2D convolutions with 3D spatial convolutions, e.g. turning 3x3 2D convolutions into a 1x3x3 convolution (the first axis indexes video frames, the second and third index the spatial height and width).
-   Each spatial attention block remains as attention over space, where the temporal dimension is treated as part of the batch axis.
-   After each spatial attention block, inserting a temporal attention block that performs attention over the first axis and treats the spatial axes as batch axes. 
-   Relative position embeddings are used in each temporal attention block so that the network can distinguish ordering of frames in a way that does not require an absolute notion of video time. 

## Configuration File

The configuration file is located in [Video Diffusion Models](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/video_diffusion_models.yaml).

## Training

First, ensure that all of the requirements are installed following the instructions [here](https://github.com/swookey-thinky/video_diffusion?tab=readme-ov-file#requirements).

To train the model, run the following from the root of the repository:

```
> python training/moving_mnist/train.py --config_path configs/moving_mnist/video_diffusion_models.yaml --num_training_steps 100000 --batch_size 8 --save_and_sample_every_n 10000
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 8.

## Sampling

To unconditionally sample from a pretrained checkpoint, you can run:

```
> python training/moving_mnist/sample.py --config_path configs/moving_mnist/video_diffusion_models.yaml --num_samples 16 --checkpoint output/moving_mnist/video_diffusion_models/diffusion-100000.pt
```

Output will be saved to the `output/moving_mnist/sample/video_diffusion_models` directory.

## Results and Checkpoints

The following results were generated using this [commit hash](https://github.com/swookey-thinky/video_diffusion/commit/fb739d9314a6bce1665d27e4f110c239183df288), after training on a single T4 instance for 100k steps at batch size 8:

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/video_diffusion_models.yaml) | [google drive](https://drive.google.com/file/d/1gAMyfBjr47sscPGNlzsxAHJegHv-dLrc/view?usp=sharing) | ![Video Diffusion Models](https://drive.google.com/uc?export=view&id=1aYxiwkgdAd6oFpXMwQhDwfiXXYJlDDFG)

## Other Resources

The authors did not release their source code, but there is another PyTorch repository from [Phil Wang](https://github.com/lucidrains) [here](https://github.com/lucidrains/video-diffusion-pytorch) if you want to look at another approach.