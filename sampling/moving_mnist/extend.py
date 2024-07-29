from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
from pathlib import Path
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from tqdm import tqdm
from typing import List, Optional

from video_diffusion.datasets.moving_mnist import MovingMNIST
from video_diffusion.diffusion import DiffusionModel
from video_diffusion.ddpm import GaussianDiffusion_DDPM
from video_diffusion.samplers import ddim, ancestral, base
from video_diffusion.utils import (
    cycle,
    load_yaml,
    DotConfig,
    normalize_to_neg_one_to_one,
    video_tensor_to_gif,
)

OUTPUT_NAME = "output/moving_mnist/extend"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sample_model(
    config_path: str,
    num_samples: int,
    guidance: float,
    checkpoint_path: str,
    sampler: str,
    reconstruction_guidance: bool,
    num_frame_overlap: int,
    source_video_path: str,
):
    global OUTPUT_NAME
    OUTPUT_NAME = f"{OUTPUT_NAME}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the Moving MNIST dataset.
    diffusion_model = GaussianDiffusion_DDPM(config=config)

    # Load the model weights if we have them
    if checkpoint_path:
        diffusion_model.load_checkpoint(checkpoint_path)

    # Build context to display the model summary.
    diffusion_model.print_model_summary()

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)

    if sampler == "ddim":
        sampler = ddim.DDIMSampler()
    elif sampler == "ancestral":
        sampler = ancestral.AncestralSampler(
            reconstruction_guidance=reconstruction_guidance
        )
    else:
        raise NotImplemented(f"Sampler {sampler} not implemented.")

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        config=config,
        num_samples=num_samples,
        sampler=sampler,
        accelerator=accelerator,
        num_frame_overlap=num_frame_overlap,
        source_video_path=source_video_path,
    )


def sample(
    diffusion_model: DiffusionModel,
    config: DotConfig,
    sampler: base.ReverseProcessSampler,
    accelerator: Accelerator,
    num_samples: int = 64,
    num_frame_overlap: int = 4,
    source_video_path: str = "",
):
    device = next(diffusion_model.parameters()).device

    context = {}
    if "super_resolution" in config:
        assert False, "Not supported yet."

    context["num_frame_overlap"] = num_frame_overlap

    # Create the source video we are extending, or grab a random one.
    if source_video_path:
        raise NotImplementedError()
    else:
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
        dataloader = DataLoader(
            dataset, batch_size=num_samples, shuffle=True, num_workers=4
        )
        dataloader = accelerator.prepare(dataloader)

        # Sample a batch of videos from the dataset
        source_videos, classes = next(iter(dataloader))

    # Trim the number of frames to the model input.
    B, C, F, H, W = source_videos.shape
    videos = source_videos[:, :, (F - config.data.input_number_of_frames) :, :, :]
    assert videos.shape[2] == config.data.input_number_of_frames

    prompts = convert_labels_to_prompts(classes)
    context["text_prompts"] = prompts
    context["classes"] = classes
    context["x_a"] = normalize_to_neg_one_to_one(videos).requires_grad_(True)

    # Save the guidance video
    video_tensor_to_gif(
        videos,
        str(f"{OUTPUT_NAME}/reconstruction_guidance.gif"),
    )

    # Save the first frame
    utils.save_image(
        videos[:, :, 0, :, :],
        str(f"{OUTPUT_NAME}/reconstruction_guidance_first_frame.png"),
        nrow=int(math.sqrt(num_samples)),
    )

    # Save the source video
    video_tensor_to_gif(
        source_videos,
        str(f"{OUTPUT_NAME}/source_videos.gif"),
    )

    # Sample from the model to check the quality.
    samples, _ = diffusion_model.sample(
        num_samples=num_samples,
        context=context,
        sampler=sampler,
    )

    # Save the first frame
    utils.save_image(
        samples[:, :, 0, :, :],
        str(f"{OUTPUT_NAME}/first_frame.png"),
        nrow=int(math.sqrt(num_samples)),
    )

    # Save the samples into an image grid
    video_tensor_to_gif(
        samples,
        str(f"{OUTPUT_NAME}/sample.gif"),
    )

    # Add the samples to the end (overlapping the last four frames)
    # of the source samples.
    extended_samples = torch.cat(
        [source_videos[:, :, :-num_frame_overlap, :, :], samples], dim=2
    )
    video_tensor_to_gif(
        extended_samples,
        str(f"{OUTPUT_NAME}/extended_sample.gif"),
    )

    # Save the prompts that were used
    with open(f"{OUTPUT_NAME}/sample.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{context['text_prompts'][i]} ")


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
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source_video", type=str, default="")
    parser.add_argument("--sampler", type=str, default="ancestral")
    parser.add_argument("--reconstruction_guidance", action="store_true")
    parser.add_argument("--num_frame_overlap", type=int, default=4)
    args = parser.parse_args()

    sample_model(
        config_path=args.config_path,
        num_samples=args.num_samples,
        guidance=args.guidance,
        checkpoint_path=args.checkpoint,
        sampler=args.sampler,
        reconstruction_guidance=args.reconstruction_guidance,
        num_frame_overlap=args.num_frame_overlap,
        source_video_path=args.source_video,
    )


if __name__ == "__main__":
    main()
