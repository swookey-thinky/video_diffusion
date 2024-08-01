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
from video_diffusion.samplers import ddim, ancestral, base, schemes
from video_diffusion.utils import (
    cycle,
    instantiate_from_config,
    load_yaml,
    DotConfig,
    normalize_to_neg_one_to_one,
    video_tensor_to_gif,
)

OUTPUT_NAME = "output/moving_mnist/sample"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sample_model(
    config_path: str,
    num_samples: int,
    guidance: float,
    checkpoint_path: str,
    sampler: str,
    sampling_scheme_path: str,
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
        sampler = ancestral.AncestralSampler()
    else:
        raise NotImplemented(f"Sampler {sampler} not implemented.")

    # Create the sampling scheme, if we have it
    sampling_scheme = None
    if sampling_scheme_path:
        sampling_scheme_config = load_yaml(sampling_scheme_path)
        sampling_scheme = instantiate_from_config(
            sampling_scheme_config.sampling_scheme.to_dict()
        )

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        config=config,
        num_samples=num_samples,
        sampler=sampler,
        sampling_scheme=sampling_scheme,
    )


def sample(
    diffusion_model: DiffusionModel,
    config: DotConfig,
    sampler: base.ReverseProcessSampler,
    num_samples: int = 64,
    sampling_scheme: Optional[schemes.SamplingSchemeBase] = None,
):
    device = next(diffusion_model.parameters()).device

    context = {}
    if "super_resolution" in config:
        assert False, "Not supported yet."

    # Sample from the model to check the quality.
    classes = torch.randint(
        0, config.data.num_classes, size=(num_samples, 2), device=device
    )
    prompts = convert_labels_to_prompts(classes)
    context["text_prompts"] = prompts
    context["classes"] = classes

    if sampling_scheme is not None:
        # Starts unconditionally
        assert sampling_scheme.num_observations == 0
        frame_indices_iterator = iter(sampling_scheme)

        # Unconditional conditioning for the first step. Note that
        # this is (B, T, C, H, W), not the normal channels first.
        samples = torch.zeros(
            size=(
                num_samples,
                sampling_scheme.video_length,
                config.data.num_channels,
                config.data.image_size,
                config.data.image_size,
            ),
        )
        step = 0
        while True:
            # ignored for non-adaptive sampling schemes
            frame_indices_iterator.set_videos(samples.to(device))
            try:
                obs_frame_indices, latent_frame_indices = next(frame_indices_iterator)
            except StopIteration:
                break

            frame_indices = torch.cat(
                [torch.tensor(obs_frame_indices), torch.tensor(latent_frame_indices)],
                dim=1,
            ).long()
            x0 = torch.stack(
                [samples[i, fi] for i, fi in enumerate(frame_indices)], dim=0
            ).clone()
            obs_mask = (
                torch.cat(
                    [
                        torch.ones_like(torch.tensor(obs_frame_indices)),
                        torch.zeros_like(torch.tensor(latent_frame_indices)),
                    ],
                    dim=1,
                )
                .view(num_samples, -1, 1, 1, 1)
                .float()
            )
            latent_mask = 1 - obs_mask

            # Move tensors to the correct device
            x0, observed_mask, latent_mask, frame_indices = (
                t.to(device) for t in [x0, obs_mask, latent_mask, frame_indices]
            )

            # Run the network. Ensure everything is still channels first
            context["x0"] = normalize_to_neg_one_to_one(x0.permute(0, 2, 1, 3, 4))
            context["frame_indices"] = frame_indices
            context["observed_mask"] = observed_mask.permute(0, 2, 1, 3, 4)
            context["latent_mask"] = latent_mask.permute(0, 2, 1, 3, 4)

            local_samples, _ = diffusion_model.sample(
                num_samples=num_samples, context=context, sampler=sampler
            )
            local_samples = local_samples.permute(0, 2, 1, 3, 4)
            for i, li in enumerate(latent_frame_indices):
                samples[i, li] = local_samples[i, -len(li) :].cpu()

            video_tensor_to_gif(
                samples.permute(0, 2, 1, 3, 4),
                str(f"{OUTPUT_NAME}/sample_step_{step}.gif"),
            )
            step += 1

        # Make samples channels first again
        samples = samples.permute(0, 2, 1, 3, 4)
    else:
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
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--sampler", type=str, default="ancestral")
    parser.add_argument("--sampling_scheme_path", type=str, default="")
    args = parser.parse_args()

    sample_model(
        config_path=args.config_path,
        num_samples=args.num_samples,
        guidance=args.guidance,
        checkpoint_path=args.checkpoint,
        sampler=args.sampler,
        sampling_scheme_path=args.sampling_scheme_path,
    )


if __name__ == "__main__":
    main()
