"""
dataloader.py
=============
PyTorch Dataset and DataLoader for the preprocessed HumanAct12 dataset.

Returns batches of:
    motion  : Tensor [B, 64, 13, 2]  – normalised 2D keypoint sequences
    label   : Tensor [B]              – integer class index (0-3)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from motion_generator.datapreprocess import (
    OUTPUT_PATH  as DEFAULT_NPZ_PATH,
    ACTION_NAMES,
    NUM_ACTIONS,
    SEQ_LEN,
    NUM_JOINTS,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HumanAct12Dataset(Dataset):
    """
    Loads the preprocessed .npz produced by datapreprocess.py.

    Args:
        npz_path  : path to dataset_preprocessed.npz
        augment   : if True, apply random horizontal flip during training
    """

    def __init__(self, npz_path=DEFAULT_NPZ_PATH, augment=False):
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"Preprocessed dataset not found at: {npz_path}\n"
                "Run  python -m motion_generator.datapreprocess  first."
            )

        data = np.load(npz_path, allow_pickle=True)
        self.motions      = data["motions"]       # [N, 64, 13, 2]  float32
        self.labels       = data["labels"]        # [N]             int64
        self.action_names = list(data["action_names"])
        self.augment      = augment

        assert self.motions.shape[1] == SEQ_LEN,   "Sequence length mismatch."
        assert self.motions.shape[2] == NUM_JOINTS, "Joint count mismatch."
        assert self.motions.shape[3] == 2,          "Expected 2D keypoints."

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        motion = self.motions[idx].copy()   # [64, 13, 2]
        label  = int(self.labels[idx])

        # Random horizontal flip augmentation (flip x -> -x because centered at 0,0)
        if self.augment and np.random.rand() < 0.5:
            motion[:, :, 0] = -motion[:, :, 0]

        motion_t = torch.tensor(motion, dtype=torch.float32)  # [64, 13, 2]
        label_t  = torch.tensor(label,  dtype=torch.long)
        return motion_t, label_t

    # ------------------------------------------------------------------
    def get_label_name(self, label_idx):
        return self.action_names[label_idx]

    def class_distribution(self):
        counts = {}
        for name in self.action_names:
            counts[name] = 0
        for lbl in self.labels:
            counts[self.action_names[int(lbl)]] += 1
        return counts


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    npz_path      = DEFAULT_NPZ_PATH,
    batch_size    = 32,
    val_split     = 0.15,
    num_workers   = 0,
    seed          = 42,
):
    """
    Build train and validation DataLoaders from the preprocessed .npz.

    Args:
        npz_path    : path to dataset_preprocessed.npz
        batch_size  : samples per batch
        val_split   : fraction of data to use for validation
        num_workers : DataLoader worker processes (0 = main process)
        seed        : random seed for reproducible split

    Returns:
        train_loader, val_loader  (torch.utils.data.DataLoader)
        dataset                   (HumanAct12Dataset – full, for metadata)
    """
    # Full dataset (no augmentation for stats / val)
    full_ds = HumanAct12Dataset(npz_path, augment=False)
    N       = len(full_ds)

    val_size   = max(1, int(N * val_split))
    train_size = N - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                    generator=generator)

    # Enable augmentation only on the training split
    # (Wrap train indices with augmented dataset)
    train_ds_aug = _AugmentedSubset(full_ds, train_ds.indices)

    train_loader = DataLoader(
        train_ds_aug,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    print(f"Dataset split  -> train: {train_size}  val: {val_size}")
    print(f"Class dist.   : {full_ds.class_distribution()}")
    return train_loader, val_loader, full_ds


class _AugmentedSubset(Dataset):
    """Thin wrapper that enables augmentation on a subset of a HumanAct12Dataset."""
    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        motion = self.base_ds.motions[self.indices[idx]].copy()
        label  = int(self.base_ds.labels[self.indices[idx]])

        # Horizontal flip augmentation
        if np.random.rand() < 0.5:
            motion[:, :, 0] = 1.0 - motion[:, :, 0]

        motion_t = torch.tensor(motion, dtype=torch.float32)
        label_t  = torch.tensor(label,  dtype=torch.long)
        return motion_t, label_t
