# Make-A-Video: Text-to-Video Generation without Text-Video Data

In this example we introduce Make-A-Video, from the paper [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792).

## Introduction

Make-A-Video improves over [Video Diffusion Models](https://arxiv.org/abs/2204.03458) through several architectural improvements, designed to retain the image generation qualities of a pretrained text-to-image diffusion model while learning the temporal dependencies from unlabeled videos.

- Fine tune an existing T2I diffusion model for video generation, using factorized spatiotemporal convolutional layers and attention
- Frame interpolation model for generating higher frame rate data
- Video super-resolution networks for generating high resolution video data.

The original paper operated in the latent space (using text embeddings to generate image embeddings), and a decoder network to generate low resolution images from the prior-generated image embeddings. here, we will generate low-resolution images in pixel space (at 32x32) and use the super-resolution models to upsample to 64x64.

### Pseudo-3D Convolutional Layers

To factorize the space and time convolutional layers, the authors introduce a Pseudo-3D convolutional layer which first performs a 2D convolution over `(B F) C H W`, followed by a 1D convolution over the temporal dimension `(B H W) C F`. The 2D convolution is initialized using the weights of the pretrained T2I model, and the temporal attention layer is initialized to the identity function.

### Pseudo-3D Attention Layers

Similar to the Pseudo-3D convolutional layer, the 3D attention layer is simlarly decomposed
a pretrained 2D attention layer, followed by a 1D attention layer over the temporal dimension
that is initialized to the identity function.

## Configuration File

The configuration file is located in [Make-A-Video](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/make_a_video.yaml).

## Training

### Pre-Training
Training the Make-A-Video model requires several steps. First, we need a pretrained image diffusion model. We could take any of the pretrained image diffusion models trained on MNIST from [Image Diffusion](https://github.com/swookey-thinky/image_diffusion), but that is less than ideal since MNIST and Moving MNIST are pretty different in terms of digit size and number of digits. So instead, we'll train a new image diffusion model from scratch on single frames from the Moving MNIST dataset. Note that for text guided diffusion, we convert the class labels of the digits in each video into text prompts of the form "<first digit> and <second digit>", so a text prompt for class labels 2 and 4 could look like one of the following: `"2 and 4"` or `"two and 4"` or `"two and four"` or `"2 and four"`.

We will use a basic v-prediction, continuous time text to image diffusion model using CLIP embeddings. The configuration for this base image diffusion model is [here](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/ddpm_32x32_v_continuous_clip.yaml).

```
> python training/moving_mnist/train_image.py --config_path configs/moving_mnist/ddpm_32x32_v_continuous_clip.yaml --num_training_steps 20000 --batch_size 128 --save_and_sample_every_n 10000
```

The above model was trained for 20k steps on a single T4 instance at batch size 128. The model wasn't fully converged by then but we found the text guidance to be good enough so we left it there. 

### Video Training

To train the video diffusion model from the pretrained weights in the first step, use:

```
> python training/moving_mnist/train.py --config_path configs/moving_mnist/make_a_video.yaml --batch_size 8 --load_model_weights_from_checkpoint output/moving_mnist_image/ddpm_32x32_v_continuous_clip/diffusion-20000.pt --num_training_steps 100000 --save_and_sample_every_n 10000
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 8.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python training/moving_mnist/sample.py --config_path configs/moving_mnist/make_a_video.yaml --num_samples 16 --checkpoint output/moving_mnist/make_a_video/diffusion-100000.pt
```

Output will be saved to the `output/moving_mnist/sample/make_a_video` directory.

## Results and Checkpoints

### Pretrained Image Diffusion Model

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/ddpm_32x32_v_continuous_clip.yaml) | [google drive](https://drive.google.com/file/d/18dw8zw63o9pI9FRPqnDVEdYXINBs_itV/view?usp=sharing) | ![DDPM](https://drive.google.com/uc?export=view&id=1hyWEkqtHfMSN0g8F0uRcubixIcB9BXCJ)


The text prompts used in the above results were:

```
4 and eight, 0 and two,     1 and zero,  8 and zero,     5 and 5,         7 and one,     2 and 2,        three and eight 
six and 8,   6 and 4,       zero and 2,  5 and nine,     1 and four,      six and 4,     0 and zero,     seven and 9 
nine and 7,  seven and 0,   1 and seven, 1 and 1,        5 and 2,         3 and two,     four and 4,     9 and 1 
6 and three, nine and 4,    seven and 8, eight and 5,    4 and eight,     5 and eight,   6 and 0,        6 and seven 
2 and 5,     one and three, 4 and nine,  three and nine, 4 and two,       three and 2,   3 and 9,        six and two 
seven and 6, 2 and 8,       three and 3, 3 and eight,    three and three, 4 and eight,   zero and eight, 0 and six 
8 and 4,     4 and 2,       9 and zero,  2 and zero,     two and six,     0 and zero,    six and six,    one and 8 
six and two, 8 and four,    7 and six,   1 and 3,        3 and 7,         zero and zero, 9 and zero,     six and 8 
```

### Make-A-Video

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/make_a_video.yaml) | [google drive](https://drive.google.com/file/d/1dlPlZd3K77TWv_ihmV-JuLtzSxpyMM-S/view?usp=sharing) | ![Make-A-Video](https://drive.google.com/uc?export=view&id=1dm4H7lsliib4KW-4T4DJeiFLRi2Ph2JD)

The text prompts used in the above results were:

```
1 and 3,       6 and two, 1 and six,     9 and 5 
two and 5,     5 and 2,   1 and three,   six and 9 
9 and 0,       2 and two, two and 9,     two and five 
two and seven, 1 and 0,   six and seven, 5 and four 
```

## Other Resources

The authors did not release their source code, but there is another PyTorch repository from [Phil Wang](https://github.com/lucidrains) [here](https://github.com/lucidrains/make-a-video-pytorch) if you want to look at another approach.
