from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import utils as torchvision_utils
from torchvision.transforms import v2
from tqdm import tqdm
from typing import List

from video_diffusion.utils import load_yaml, cycle, DotConfig, video_tensor_to_gif
from video_diffusion.ddpm import GaussianDiffusion_DDPM
from video_diffusion.diffusion import DiffusionModel
from video_diffusion.datasets.moving_mnist import MovingMNIST

OUTPUT_NAME = "output/moving_mnist"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(
    num_training_steps: int,
    batch_size: int,
    config_path: str,
    save_and_sample_every_n: int,
    load_model_weights_from_checkpoint: str,
    resume_from: str,
    sample_with_guidance: bool = False,
):
    global OUTPUT_NAME
    OUTPUT_NAME = f"{OUTPUT_NAME}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Load the MNIST dataset. This is a supervised dataset so
    # it contains both images and class labels. We will ignore the class
    # labels for now.
    dataset = MovingMNIST(
        ".",
        transform=v2.Compose(
            [
                # To the memory requirements, resize the MNIST
                # images from (64,64) to (32, 32).
                v2.Resize(
                    size=(config.data.image_size, config.data.image_size),
                    antialias=True,
                ),
                # Convert the motion images to (0,1) float range
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
    )

    # Create the dataloader for the MNIST dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    assert "diffusion_cascade" not in config
    diffusion_model = GaussianDiffusion_DDPM(config=config)

    # Load the model weights if we have them
    if load_model_weights_from_checkpoint:
        diffusion_model.load_checkpoint(load_model_weights_from_checkpoint)

    if resume_from:
        diffusion_model.load_checkpoint(resume_from)

    # Build context to display the model summary.
    diffusion_model.print_model_summary()

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader = accelerator.prepare(dataloader)

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Now create the optimizer. The optimizer choice and parameters come from
    # the paper:
    # "We tried Adam [31] and RMSProp early on in our experimentation process and chose the
    #  former. We left the hyperparameters to their standard values. We set the learning
    #  rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5
    #  for the 256 × 256 images, which seemed unstable to train with the larger learning rate."
    optimizers = diffusion_model.configure_optimizers(learning_rate=2e-4)

    # Load the optimizers if we have them from the checkpoint
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu")
        num_optimizers = checkpoint["num_optimizers"]
        for i in range(num_optimizers):
            optimizers[i].load_state_dict(checkpoint["optimizer_state_dicts"][i])

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)
    for optimizer_idx in range(len(optimizers)):
        optimizers[optimizer_idx] = accelerator.prepare(optimizers[optimizer_idx])

    # Configure the learning rate schedule
    learning_rate_schedules = diffusion_model.configure_learning_rate_schedule(
        optimizers
    )
    for schedule_idx in range(len(learning_rate_schedules)):
        learning_rate_schedules[schedule_idx] = accelerator.prepare(
            learning_rate_schedules[schedule_idx]
        )

    # Step counter to keep track of training
    step = 0

    # Not mentioned in the DDPM paper, but the original implementation
    # used gradient clipping during training.
    max_grad_norm = 1.0
    average_loss = 0.0
    average_loss_cumulative = 0.0
    num_samples = 16

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, labels = next(dataloader)
            context = {"labels": labels}

            # Convert the labels to text prompts
            text_prompts = convert_labels_to_prompts(labels)
            context["text_prompts"] = text_prompts

            # Clip the input to the required number of frames.
            # Videos come in as (B, C, F, H, W).
            images = images[:, :, : config.data.input_number_of_frames, :, :]

            # Train each cascade in the model using the given data.
            stage_loss = 0
            for model_for_layer, optimizer_for_layer, schedule_for_layer in zip(
                diffusion_model.models(), optimizers, learning_rate_schedules
            ):
                # Is this a super resolution model? If it is, then generate
                # the low resolution imagery as conditioning.
                config_for_layer = model_for_layer.config()
                context_for_layer = context.copy()
                images_for_layer = images

                if "super_resolution" in config_for_layer:
                    # First create the low resolution context.
                    low_resolution_spatial_size = (
                        config_for_layer.super_resolution.low_resolution_spatial_size
                    )
                    low_resolution_images = v2.functional.resize(
                        images,
                        size=(
                            low_resolution_spatial_size,
                            low_resolution_spatial_size,
                        ),
                        antialias=True,
                    )
                    context_for_layer[
                        config_for_layer.super_resolution.conditioning_key
                    ] = low_resolution_images

                # If the images are not the right shape for the model input, then
                # we need to resize them too. This could happen if we are the intermediate
                # super resolution layers of a multi-layer cascade.
                model_input_spatial_size = config_for_layer.data.image_size

                B, C, F, H, W = images.shape
                if H != model_input_spatial_size or W != model_input_spatial_size:
                    images_for_layer = v2.functional.resize(
                        images,
                        size=(
                            model_input_spatial_size,
                            model_input_spatial_size,
                        ),
                        antialias=True,
                    )

                # Calculate the loss on the batch of training data.
                loss_dict = model_for_layer.loss_on_batch(
                    images=images_for_layer, context=context_for_layer
                )
                loss = loss_dict["loss"]

                # Calculate the gradients at each step in the network.
                accelerator.backward(loss)

                # On a multi-gpu machine or cluster, wait for all of the workers
                # to finish.
                accelerator.wait_for_everyone()

                # Clip the gradients.
                accelerator.clip_grad_norm_(
                    model_for_layer.parameters(),
                    max_grad_norm,
                )

                # Perform the gradient descent step using the optimizer.
                optimizer_for_layer.step()
                schedule_for_layer.step()

                # Resent the gradients for the next step.
                optimizer_for_layer.zero_grad()
                stage_loss += loss.item()

            # Show the current loss in the progress bar.
            stage_loss = stage_loss / len(optimizers)
            progress_bar.set_description(
                f"loss: {stage_loss:.4f} avg_loss: {average_loss:.4f}"
            )
            average_loss_cumulative += stage_loss

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                sample(
                    diffusion_model=diffusion_model,
                    step=step,
                    config=config,
                    num_samples=num_samples,
                    sample_with_guidance=sample_with_guidance,
                )
                save(diffusion_model, step, loss, optimizers, config)
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        step=step,
        config=config,
        num_samples=num_samples,
        sample_with_guidance=sample_with_guidance,
    )
    save(diffusion_model, step, loss, optimizers, config)


def sample(
    diffusion_model: DiffusionModel,
    step,
    config: DotConfig,
    num_samples=64,
    sample_with_guidance: bool = False,
):
    device = next(diffusion_model.parameters()).device

    context = {}
    if "super_resolution" in config:
        images, classes = next(iter(validation_dataloader))
        prompts = convert_labels_to_prompts(classes)
        context["text_prompts"] = prompts
        context["classes"] = classes

        # Downsample to create the low resolution context
        low_resolution_spatial_size = (
            config.super_resolution.low_resolution_spatial_size
        )
        low_resolution_images = transforms.functional.resize(
            images,
            size=(
                low_resolution_spatial_size,
                low_resolution_spatial_size,
            ),
            antialias=True,
        )
        context[config.super_resolution.conditioning_key] = low_resolution_images
    else:
        # Sample from the model to check the quality.
        classes = torch.randint(
            0, config.data.num_classes, size=(num_samples, 2), device=device
        )
        prompts = convert_labels_to_prompts(classes)
        context["text_prompts"] = prompts
        context["classes"] = classes

    if sample_with_guidance:
        for guidance in [0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0]:
            samples, intermediate_stage_output = diffusion_model.sample(
                num_samples=num_samples,
                context=context,
                classifier_free_guidance=guidance,
            )

            # Save the samples into an image grid
            video_tensor_to_gif(
                samples,
                str(f"{OUTPUT_NAME}/sample-{step}-cfg-{guidance}.gif"),
            )

            # Save the intermedidate stages if they exist
            if intermediate_stage_output is not None:
                for layer_idx, intermediate_output in enumerate(
                    intermediate_stage_output
                ):
                    video_tensor_to_gif(
                        intermediate_output,
                        str(
                            f"{OUTPUT_NAME}/sample-{step}-cfg-{guidance}-stage-{layer_idx}.gif"
                        ),
                    )

    else:
        samples, intermediate_stage_output = diffusion_model.sample(
            num_samples=num_samples, context=context
        )

        # Save the first frame
        torchvision_utils.save_image(
            samples[:, :, 0, :, :],
            str(f"{OUTPUT_NAME}/sample-{step}.png"),
            nrow=int(math.sqrt(num_samples)),
        )

        # Save the samples into an image grid
        video_tensor_to_gif(
            samples,
            str(f"{OUTPUT_NAME}/sample-{step}.gif"),
        )

        # Save the intermedidate stages if they exist
        if intermediate_stage_output is not None:
            for layer_idx, intermediate_output in enumerate(intermediate_stage_output):
                video_tensor_to_gif(
                    intermediate_output,
                    str(f"{OUTPUT_NAME}/sample-{step}-stage-{layer_idx}.gif"),
                )

    # Save the prompts that were used
    with open(f"{OUTPUT_NAME}/sample-{step}.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{context['text_prompts'][i]} ")

    # Save the low-resolution imagery if it was used.
    if "super_resolution" in config:
        video_tensor_to_gif(
            context[config.super_resolution.conditioning_key],
            str(f"{OUTPUT_NAME}/low_resolution_context-{step}.gif"),
        )


def save(
    diffusion_model,
    step,
    loss,
    optimizers: List[torch.optim.Optimizer],
    config: DotConfig,
):
    # Save a corresponding model checkpoint.
    torch.save(
        {
            "step": step,
            "model_state_dict": diffusion_model.state_dict(),
            "num_optimizers": len(optimizers),
            "optimizer_state_dicts": [
                optimizer.state_dict() for optimizer in optimizers
            ],
            "loss": loss,
            "config": config.to_dict(),
        },
        f"{OUTPUT_NAME}/diffusion-{step}.pt",
    )


def convert_labels_to_prompts(labels: torch.Tensor) -> List[str]:
    """Converts MNIST class labels to text prompts.

    Supports both the strings "0" and "zero" to describe the
    class labels.
    """
    # The conditioning we pass to the model will be a vectorized-form of
    # MNIST classes. Since we have a fixed number of classes, we can create
    # a hard-coded "embedding" of the MNIST class label.
    text_labels = [
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
    ]

    # First convert the labels into a list of string prompts
    prompts = [
        f"{text_labels[labels[i][0]][torch.randint(0, len(text_labels[labels[i][0]]), size=())]} and {text_labels[labels[i][1]][torch.randint(0, len(text_labels[labels[i][1]]), size=())]}"
        for i in range(labels.shape[0])
    ]
    return prompts


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--config_path", type=str, default="configs/glide.yaml")
    parser.add_argument("--save_and_sample_every_n", type=int, default=100)
    parser.add_argument("--load_model_weights_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")
    args = parser.parse_args()

    train(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
        save_and_sample_every_n=args.save_and_sample_every_n,
        load_model_weights_from_checkpoint=args.load_model_weights_from_checkpoint,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
