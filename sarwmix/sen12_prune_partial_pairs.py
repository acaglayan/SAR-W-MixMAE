# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import os
import numpy as np
import shutil
import argparse
from pathlib import Path


def channel_zero_fractions(img, zero_val=0, atol=0.0):
    """
    img: shape (C,H,W) or (H,W) where C=2 for VV,VH
    returns: per-channel zero fractions as a 1D array
    """
    if img.ndim == 2:  # single channel
        z = np.isclose(img, zero_val, atol=atol) if atol > 0 else (img == zero_val)
        return np.array([z.mean()], dtype=np.float32)
    elif img.ndim == 3:  # C,H,W
        if atol > 0:
            z = np.isclose(img, zero_val, atol=atol)
        else:
            z = (img == zero_val)
        C = img.shape[0]
        return z.reshape(C, -1).mean(axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

def pair_is_partial(img1, img2, threshold=0.25, zero_val=0, atol=0.0):
    """
    True if ANY channel in either timepoint has zero_fraction >= threshold.
    """
    f1 = channel_zero_fractions(img1, zero_val=zero_val, atol=atol)
    f2 = channel_zero_fractions(img2, zero_val=zero_val, atol=atol)
    return (f1 >= threshold).any() or (f2 >= threshold).any(), f1, f2

def filter_and_copy_dataset(src_folder, dest_folder, log_file_path, threshold=0.25, zero_val=0, atol=0.0):
    os.makedirs(dest_folder, exist_ok=True)
    with open(log_file_path, 'w') as log_file:
        log_file.write("Skipped files (>= threshold zero fraction in any VV/VH):\n")
        total_files = valid_files = skipped_files = 0

        # use os.walk in case you have nested dirs
        for root, _, files in os.walk(src_folder):
            for filename in files:
                if not filename.endswith(".npy"):
                    continue
                total_files += 1
                file_path = os.path.join(root, filename)

                data = np.load(file_path, mmap_mode='r')  # safe on memory
                img1, img2 = data[0], data[1]            # each (2,H,W) = (VV,VH)

                is_partial, f1, f2 = pair_is_partial(img1, img2, threshold, zero_val, atol)

                if not is_partial:
                    rel = os.path.relpath(file_path, src_folder)
                    out_path = os.path.join(dest_folder, rel)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    shutil.copy(file_path, out_path)
                    valid_files += 1
                else:
                    # log per-channel fractions for debugging (VV, VH)
                    log_file.write(f"{os.path.relpath(file_path, src_folder)} "
                                   f"img1(VV,VH)={tuple(np.round(f1,3))} "
                                   f"img2(VV,VH)={tuple(np.round(f2,3))}\n")
                    skipped_files += 1

        print(f"Processed {total_files} files in {src_folder}.")
        print(f"Copied {valid_files} valid files to {dest_folder}.")
        print(f"Skipped {skipped_files} files. Details logged in {log_file_path}.")


def create_clean_dataset(train_folder, test_folder, output_folder):
    """
    Creates a clean dataset by removing `.npy` files with zeros and copying valid files to new train/test folders.

    Args:
        train_folder (str): Path to the original training dataset folder.
        test_folder (str): Path to the original testing dataset folder.
        output_folder (str): Path to the base output folder for the clean dataset.
    """
    # Paths for clean train and test datasets
    clean_train_folder = os.path.join(output_folder, "train")
    clean_test_folder = os.path.join(output_folder, "test")

    # Log file paths
    train_log_file = os.path.join(output_folder, "train_skipped_files.log")
    test_log_file = os.path.join(output_folder, "test_skipped_files.log")

    # Process train and test datasets
    print("Processing train dataset...")
    filter_and_copy_dataset(train_folder, clean_train_folder, train_log_file)

    print("Processing test dataset...")
    filter_and_copy_dataset(test_folder, clean_test_folder, test_log_file)

    print("Clean dataset created successfully!")
    print(f"Skipped files are logged in:\n - {train_log_file}\n - {test_log_file}")


if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Create clean and aligned SEN12FLOOD paired dataset.")
    parser.add_argument("--dataset-root", required=True, type=str, help="Path to SEN12FLOOD ALIGNED and PAIRED dataset root.")
    parser.add_argument("--save-root", required=True, type=str, help="Path to save the curated paired data (train/test).")
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    train_folder_path = Path(dataset_root) / "train"
    test_folder_path = Path(dataset_root) / "test"
    output_folder_path = args.save_root

    create_clean_dataset(train_folder_path, test_folder_path, output_folder_path)
