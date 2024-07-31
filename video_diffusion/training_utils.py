"""Utilities for training."""

import numpy as np
import torch
from typing import Tuple


def sample_masks_for_training_batch(
    video_batch: torch.Tensor, max_frames: int, method: str = "uniform"
):
    """Samples masks (observations and latents) for flexible training.

    Creates a tensor batch of data used for training, following the mask and
    batch creation scheme of "Flexible Diffusion Modeling of Long Videos"
    (https://arxiv.org/abs/2205.11495).

    For example, assume the video_batch is a tensor batch of data of shape (B,C,F,H,W),
    e.g. video frame data of 16 frames. To perform unconditional modeling of this video dataset,
    frame indices would be a tensor batch where each row is range(0,F) - all frames are
    included. observation_mask would be zeros of shape (B,C,F,1,1) - indicating no frames
    are used for conditioning. And latent_mask would be ones of shape (B,C,F,1,1) - all frames
    are used for generation.

    To train a conditional video model, conditioned on a image (single video frame),
    the video batch is the same tensor batch of shape (B,C,F,H,W). The frame indices
    would be a tensor batch where each row is range(1,F). The observation mask would
    be all ones for the first frame and zeros elsewhere. And the latents mask would
    would be all zeros for the first frame and all ones elsewhere.

    In this function, we are sampling *random* masks, so that at inference time
    we can generalize to any arbitrary sampling scheme.

    Args:
        video_batch: Tensor batch of video data
        max_frames: The maximum frames in the output video
        method: Sampling method. Can be uniform (all frames are used, no observed frames)
          or random (random observations/conditining).

    Returns:
        Tuple of:
            batch: Tensor batch of training data
            frame_indices: Tensor batch of frame indices, indicating the frames to
                use for training (latent frames)
            observation_mask: Tensor batch of masks, where 1 indicates an observed
                frame (e.g. a frame that is used or conditioning).
            latent_mask: Tensor batch of masks, where 1 indicates a latent mask
                (e.g. a frame used for noising and training).
    """
    B, C, T, H, W = video_batch.shape
    N = max_frames

    if method == "uniform":
        video_batch = video_batch[:, :, :max_frames, :, :]
        frame_indices = torch.tile(
            torch.arange(end=N, device=video_batch.device)[None, ...],
            (B, 1),
        )
        observed_mask = torch.zeros(
            size=(B, C, N, 1, 1), dtype=torch.float32, device=video_batch.device
        )
        latent_mask = torch.ones(
            size=(B, C, N, 1, 1), dtype=torch.float32, device=video_batch.device
        )
        return video_batch, frame_indices, observed_mask, latent_mask

    # Following code assumes shapes of (B, T, C, H, W)
    video_batch = video_batch.permute(0, 2, 1, 3, 4)

    # Base mask, of shape (B, 1, T, 1, 1) - all frames
    masks = {
        k: torch.zeros_like(video_batch[:, :, :1, :1, :1]) for k in ["obs", "latent"]
    }

    for obs_row, latent_row in zip(*[masks[k] for k in ["obs", "latent"]]):
        # First sample some frames to use for training/generation (latent frames).
        latent_row[_sample_some_indices(max_indices=N, T=T)] = 1.0

        # From the frames that are left, use some for conditioning (observed frames)
        # or for training (latent frames).
        while True:
            # Select whether we are adding a latent or observed mask
            mask = obs_row if torch.rand(()) < 0.5 else latent_row
            # Grab some random indices for this mask (latent or observed)
            indices = torch.tensor(_sample_some_indices(max_indices=N, T=T))
            taken = (obs_row[indices] + latent_row[indices]).view(-1)
            # Remove indices that are already used in a mask
            indices = indices[taken == 0]
            # If we have used all of the frames, break.
            if len(indices) > N - sum(obs_row) - sum(latent_row):
                break
            # Otherwise update the indices
            mask[indices] = 1.0

    any_mask = (masks["obs"] + masks["latent"]).clip(max=1)
    batch, (obs_mask, latent_mask), frame_indices = _prepare_training_batch(
        any_mask, video_batch, (masks["obs"], masks["latent"]), max_frames=N
    )

    # Flip back to expected ordering of (B, C, T, H, W)
    return (
        batch.permute(0, 2, 1, 3, 4),
        frame_indices,
        obs_mask.permute(0, 2, 1, 3, 4),
        latent_mask.permute(0, 2, 1, 3, 4),
    )


def _sample_some_indices(max_indices: int, T: int):
    s = torch.randint(low=1, high=max_indices + 1, size=())
    max_scale = T / (s - 0.999)
    scale = np.exp(np.random.rand() * np.log(max_scale))
    pos = torch.rand(()) * (T - scale * (s - 1))
    indices = [int(pos + i * scale) for i in range(s)]
    # do some recursion if we have somehow failed to satisfy the consrtaints
    if all(i < T and i >= 0 for i in indices):
        return indices
    else:
        print(
            "warning: sampled invalid indices",
            [int(pos + i * scale) for i in range(s)],
            "trying again",
        )
        return _sample_some_indices(max_indices, T)


def _prepare_training_batch(
    mask,
    video_batch,
    tensors,
    max_frames: int,
    pad_with_random_frames: bool = True,
):
    """
    Prepare training batch by selecting frames from batch1 according to mask,
    appending uniformly sampled frames from batch2, and selecting the corresponding
    elements from tensors (usually obs_mask and latent_mask).
    """
    B, T, *_ = mask.shape

    # Remove unit C, H, W dims
    mask = mask.view(B, T)

    # T is the number of frames in the source video, and can be longer
    # than max_frames. effective_T is max_frames since we are always padding the batch
    # to fill it out.
    effective_T = max_frames if pad_with_random_frames else mask.sum(dim=1).max().int()

    indices = torch.zeros_like(mask[:, :effective_T], dtype=torch.int64)
    new_batch = torch.zeros_like(video_batch[:, :effective_T])
    new_tensors = [torch.zeros_like(t[:, :effective_T]) for t in tensors]

    for b in range(B):
        instance_T = mask[b].sum().int()
        indices[b, :instance_T] = mask[b].nonzero().flatten()
        indices[b, instance_T:] = (
            torch.randint_like(indices[b, instance_T:], high=T)
            if pad_with_random_frames
            else 0
        )
        new_batch[b, :instance_T] = video_batch[b][mask[b] == 1]
        new_batch[b, instance_T:] = video_batch[b][indices[b, instance_T:]]
        for new_t, t in zip(new_tensors, tensors):
            new_t[b, :instance_T] = t[b][mask[b] == 1]
            new_t[b, instance_T:] = t[b][indices[b, instance_T:]]

    return (new_batch, new_tensors, indices)
