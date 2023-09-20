# Copyright 2023 by Ismail Khalfaoui-Hassani, ANITI Toulouse.
#
# All rights reserved.
#
# This file is part of the Dcls-Audio package, and
# is released under the "MIT License Agreement".
# Please see the LICENSE file that should have been included as part
# of this package.

import h5py
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import AmplitudeToDB, MelScale, Spectrogram
from torchvision import transforms

# Calculated on the balanced_train subset
AUDIOSET_DEFAULT_MEAN = -18.2696
AUDIOSET_DEFAULT_STD = 30.5735
from timm.data.random_erasing import RandomErasing

import augmentations


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == "AUDIOSET":
        if is_train:
            dataset = AudioSetDataset(
                args.data_path, "full_unbal_bal_train_wav.h5", transform=transform
            )
        else:
            dataset = AudioSetDataset(args.data_path, "eval.h5", transform=transform)
        nb_classes = 527
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    sample_rate = 32000
    window_size = 1024
    hop_length = 320
    n_mels = 128
    f_min = 50
    f_max = 14000

    t = [augmentations.PadOrTruncate(10 * sample_rate)]

    if is_train:
        t.extend(
            [
                augmentations.RandomRoll(dims=(1,)),
                augmentations.SpeedPerturbation(rates=(0.5, 1.5), p=0.5),
            ]
        )

    t.append(Spectrogram(n_fft=window_size, hop_length=hop_length, power=2))

    if is_train:
        t.append(
            RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.resplit,
                device="cpu",
            )
        )

    t.extend(
        [
            MelScale(
                n_mels=n_mels,
                sample_rate=sample_rate,
                f_min=f_min,
                f_max=f_max,
                n_stft=window_size // 2 + 1,
            ),
            AmplitudeToDB(),
        ]
    )
    t.append(transforms.Normalize((AUDIOSET_DEFAULT_MEAN,), (AUDIOSET_DEFAULT_STD,)))

    return transforms.Compose(t)


class AudioSetDataset(Dataset):
    def __init__(
        self, root, split_name, transform=None, target_transform=None, download=False
    ):

        self.sample_rate = 32000
        self.clip_length = 10
        self.split_name = split_name
        self.hdf5_file = root + split_name
        with h5py.File(self.hdf5_file, "r") as f:
            self.length = len(f["audio_name"])
            print(f"Dataset from {self.hdf5_file} with length {self.length}.")
        self.dataset_file = None  # lazy init
        self.clip_length = self.clip_length * self.sample_rate
        self.transform = transform
        self.target_transform = target_transform

    def open_hdf5(self):
        self.dataset_file = h5py.File(self.hdf5_file, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.

        Args:
          meta: {
            'hdf5_path': str,
            'index_in_hdf5': int}
        Returns:
          data_dict: {
            #'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        if self.dataset_file is None:
            self.open_hdf5()

        # audio_name = self.dataset_file['audio_name'][index].decode()
        waveform = (
            torch.tensor(self.dataset_file["waveform"][index] / (2 ** 15))
            .float()
            .unsqueeze(0)
        )

        if self.transform is not None:
            waveform = self.transform(waveform)

        target = torch.tensor(self.dataset_file["target"][index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return waveform, target.to(waveform.dtype)
