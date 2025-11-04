# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import os
import re
import csv
import numpy as np
import torch
from torch.utils.data import dataset, DataLoader
import rasterio
from tqdm import tqdm
from weighting import compute_weights


class BigEarthNetv2(dataset.Dataset):
    LABEL_MAP = {
        "Urban fabric": 0,
        "Industrial or commercial units": 1,
        "Arable land": 2,
        "Permanent crops": 3,
        "Pastures": 4,
        "Complex cultivation patterns": 5,
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
        "Agro-forestry areas": 7,
        "Broad-leaved forest": 8,
        "Coniferous forest": 9,
        "Mixed forest": 10,
        "Natural grassland and sparsely vegetated areas": 11,
        "Moors, heathland and sclerophyllous vegetation": 12,
        "Transitional woodland, shrub": 13,
        "Beaches, dunes, sands": 14,
        "Inland wetlands": 15,
        "Coastal wetlands": 16,
        "Inland waters": 17,
        "Marine waters": 18
    }

    def __init__(self, csv_path, root_dir, transform=None, weight_mode: str = "per_channel"):
        self.root_dir = root_dir
        self.transform = transform
        self.weight_mode = weight_mode
        self.data = []
        
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            next(reader)  # skip the header row
            for row in reader:
                labels, s1_name = self.parse_labels(row[1]), row[4]
                self.data.append((s1_name, labels))

    def parse_labels(self, label_str):
        label_str = label_str.strip("[]").replace("\n", "")
        label_list = re.findall(r"'(.*?)'", label_str)
        multi_hot = np.zeros(len(self.LABEL_MAP), dtype=int)
        for label in label_list:
            if label in self.LABEL_MAP:
                multi_hot[self.LABEL_MAP[label]] = 1
            else:
                raise ValueError(f"Unknown {label_str} label!")
        
        return multi_hot

    def load_sar_data(self, s1_name):
        first_folder = "_".join(s1_name.split("_")[:5])
        folder = os.path.join(self.root_dir, first_folder, s1_name)
        vh_path = os.path.join(folder, f"{s1_name}_VH.tif")
        vv_path = os.path.join(folder, f"{s1_name}_VV.tif")
        
        vh_data = self.read_geotiff(vh_path)
        vv_data = self.read_geotiff(vv_path)

        return np.stack([vh_data, vv_data], axis=0)

    def read_geotiff(self, file_path):
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
        # Validate data
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError(f"Data at {data_path} contains NaN or Inf")
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s1_name, labels = self.data[idx]
        sar_data = self.load_sar_data(s1_name)
        
        labels = torch.from_numpy(labels)
        sar_data = torch.from_numpy(sar_data)
        
        if self.transform:
            sar_data = self.transform(sar_data)
            
        mse_linear_weight = compute_weights(sar_data, mode=self.weight_mode)

        return sar_data, labels, mse_linear_weight


# ---------------------------------------------------------------------
# Utility function to calculate mean and std
# ---------------------------------------------------------------------

def calculate_mean_std(dataset, batch_size=16, num_workers=4, max_samples=None):
    """
    Compute mean and std of a dataset.
    This iterates through the dataset (or a subset if max_samples is provided),
    computing per-channel mean and std.

    Args:
        dataset (Dataset): PyTorch dataset with samples of shape [C,H,W].
        batch_size (int): Batch size for loading data.
        num_workers (int): Number of workers for the DataLoader.
        max_samples (int, optional): If provided, only process this many samples.

    Returns:
        mean (torch.Tensor): mean per channel
        std (torch.Tensor): std per channel
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    n_samples = 0
    channel_sum = 0.0
    channel_sum_sq = 0.0
    
    with tqdm(total=len(loader), desc="Calculating Mean & Std") as pbar:
        for i, (data, _, _) in enumerate(loader):
            # data shape: (B, C, H, W)
            # convert to float
            data = data.float()
            # flatten spatial dims
            B, C, H, W = data.shape
            data = data.view(B, C, -1)
            channel_sum += data.sum(dim=(0, 2))
            channel_sum_sq += (data ** 2).sum(dim=(0, 2))
            n_samples += B * H * W

            if max_samples is not None and n_samples >= max_samples:
                break
            pbar.update(1)  # update tqdm progress bar

    mean = channel_sum / n_samples
    std = torch.sqrt((channel_sum_sq / n_samples) - mean ** 2)
    return mean, std



if __name__ == "__main__":
    train_val_csv = "benv2_train_val.csv"

    data_root = "/path/datasets/bigearthnet/v2/BigEarthNet-S1"

    bigearthnetv2_mean = [-19.1966, -12.5144]
    bigearthnetv2_std = [5.4501, 5.0019]

    dataset = BigEarthNetv2(train_val_csv, data_root)
    # Calculate mean and std for normalization
    mean, std = calculate_mean_std(dataset, batch_size=256, num_workers=32)
    print("Mean:", mean)
    print("Std:", std)
