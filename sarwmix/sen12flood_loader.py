# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Custom Dataset Class for Loading Flood Data
class FloodDataset(Dataset):
    def __init__(self, root_dir, transform=None, debug_mode=0):
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.transform = transform
        # Check if debug mode is enabled (1), and take only a few samples if so
        if debug_mode:
            if "/train" in root_dir:
                self.file_list = self.file_list[:480]
            else:   # test
                self.file_list = self.file_list[:50]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)

        # Load the .npy file
        data = np.load(file_path)  # Shape: (2, 2, 512, 512)
        # Separate the two images in the pair
        img1 = data[0]  # Shape: (2, 512, 512)
        img2 = data[1]  # Shape: (2, 512, 512)

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()

        img1[img1 == 0] = 1e-6
        img1 = 10 * np.log10(img1)
        img2[img2 == 0] = 1e-6
        img2 = 10 * np.log10(img2)
        # Apply transformations if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Extract ground truth label from filename (last part after '__')
        gt_label = int(file_name.split('__')[-1].split('.')[0])
        # num_classes = 2
        # gt_label = np.zeros(num_classes, dtype=float)
        # gt_label[idx] = 1

        # Convert data and label to torch tensors
        gt_label = torch.tensor(gt_label, dtype=torch.int64)

        return img1, img2, gt_label


# Function to create data loaders
def create_dataloaders(train_dir, test_dir, batch_size=16, num_workers=4):
    # Normalization values for the dataset (VH, VV channels)
    # we use bigearthnet dataset mean and std for sen12 as we use that pretrained model
    mean, std = [-19.2309, -12.5951], [3.1672, 2.9644]

    # Transformations for the input data
    transform = transforms.Compose([
        transforms.Normalize(mean=mean, std=std),
    ])

    # Create training and test datasets
    train_dataset = FloodDataset(root_dir=train_dir, transform=transform)
    test_dataset = FloodDataset(root_dir=test_dir, transform=transform)

    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    train_dir = "/PATH_2_DATASET/SEN12FLOOD_PAIRS/train"
    test_dir = "/PATH_2_DATASET/SEN12FLOOD_PAIRS/test"

    # Create data loaders
    train_loader, test_loader = create_dataloaders(train_dir, test_dir)

    # Example usage of the data loader
    for img1, img2, label in train_loader:
        print("Img1 shape:", img1.shape)
        print("Img2 shape:", img2.shape)
        print("Label:", label)
        break
