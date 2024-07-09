"""Moving MNIST dataset from http://www.cs.toronto.edu/~nitish/unsupervised_video.

Augments the original Moving MNIST dataset with labels for text guided video diffusion.
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset


class MovingMNIST(Dataset):
    """Moving MNIST dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Download the data to the root dir if it does not exist
        from urllib.request import urlretrieve

        def download(filename, source_url):
            print(f"Downloading {source_url} to {filename}")
            urlretrieve(source_url, filename)

        videos_file_name = os.path.join(root_dir, "MovingMNIST/videos_data.npz")
        labels_file_name = os.path.join(root_dir, "MovingMNIST/labels_data.npz")

        _VIDEOS_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _LABELS_URL = "https://drive.google.com/uc?export=view&id=17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"

        if not os.path.isfile(videos_file_name):
            download(videos_file_name, _VIDEOS_URL)
        if not os.path.isfile(labels_file_name):
            download(labels_file_name, _LABELS_URL)

        self._num_frames_per_video = 30
        self._num_videos = 10000
        self._num_digits_per_video = 2

        with np.load(videos_file_name) as npz:
            videos_np = npz[npz.files[0]]

        with np.load(labels_file_name) as npz:
            labels_np = npz[npz.files[0]]

        self._video_data = torch.from_numpy(
            videos_np.reshape(self._num_videos, 1, self._num_frames_per_video, 64, 64)
        )
        self._labels_data = torch.from_numpy(
            labels_np.reshape(
                self._num_videos, self._num_frames_per_video, self._num_digits_per_video
            )[:, 0, :]
        ).squeeze()

    def __len__(self):
        return self._video_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video = self._video_data[idx]
        labels = self._labels_data[idx]
        if self.transform:
            video = self.transform(video)
        return video, labels


class MovingMNISTImage(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, train: bool = True):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Download the data to the root dir if it does not exist
        from urllib.request import urlretrieve

        def download(filename, source_url):
            print(f"Downloading {source_url} to {filename}")
            urlretrieve(source_url, filename)

        train_videos_file_name = os.path.join(root_dir, "MovingMNIST/videos_data.npz")
        train_labels_file_name = os.path.join(root_dir, "MovingMNIST/labels_data.npz")
        val_videos_file_name = os.path.join(
            root_dir, "MovingMNIST/videos_data_validation.npz"
        )
        val_labels_file_name = os.path.join(
            root_dir, "MovingMNIST/labels_data_validation.npz"
        )

        _TRAIN_VIDEOS_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _TRAIN_LABELS_URL = "https://drive.google.com/uc?export=view&id=17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"

        _VAL_VIDEOS_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _VAL_LABELS_URL = "https://drive.google.com/uc?export=view&id=17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"

        if train:
            videos_file_name = train_videos_file_name
            labels_file_name = train_labels_file_name

            _VIDEOS_URL = _TRAIN_VIDEOS_URL
            _LABELS_URL = _TRAIN_LABELS_URL
        else:
            videos_file_name = val_videos_file_name
            labels_file_name = val_labels_file_name
            _VIDEOS_URL = _VAL_VIDEOS_URL
            _LABELS_URL = _VAL_LABELS_URL

        if not os.path.isfile(videos_file_name):
            download(videos_file_name, _VIDEOS_URL)
        if not os.path.isfile(labels_file_name):
            download(labels_file_name, _LABELS_URL)

        self._num_frames_per_video = 30
        self._num_videos = 10000
        self._num_digits_per_video = 2

        with np.load(videos_file_name) as npz:
            videos_np = npz[npz.files[0]]

        with np.load(labels_file_name) as npz:
            labels_np = npz[npz.files[0]]

        # The video data is (num_videos * num_frames, 1, 64, 64)
        self._video_data = torch.from_numpy(videos_np)
        self._labels_data = torch.from_numpy(labels_np).squeeze()

    def __len__(self):
        return self._video_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video = self._video_data[idx]
        labels = self._labels_data[idx]
        if self.transform:
            video = self.transform(video)
        return video, labels
