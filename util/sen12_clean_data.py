# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import os
import numpy as np
import shutil


def check_images_for_zeros(data):
    """
    Checks if there are any zeros in the images of the data.
    data[0] -> img1 (2x512x512)
    data[1] -> img2 (2x512x512)
    Returns:
        (has_zero_img1, has_zero_img2) - Boolean values indicating zero presence in img1 and img2
    """
    img1 = data[0]
    img2 = data[1]

    has_zero_img1 = np.any(img1 == 0)
    has_zero_img2 = np.any(img2 == 0)

    return has_zero_img1, has_zero_img2


def filter_and_copy_dataset(src_folder, dest_folder, log_file_path):
    """
    Filters `.npy` files in the source folder for zero presence and copies valid files to the destination folder.

    Args:
        src_folder (str): Path to the source folder containing `.npy` files.
        dest_folder (str): Path to the destination folder to store valid files.
        log_file_path (str): Path to the log file to record skipped files.
    """
    # Ensure destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Open the log file for skipped files
    with open(log_file_path, 'w') as log_file:
        log_file.write("Skipped files due to zero presence:\n")

        # Counters for logging
        total_files = 0
        valid_files = 0
        skipped_files = 0

        for filename in os.listdir(src_folder):
            if filename.endswith(".npy"):
                total_files += 1
                file_path = os.path.join(src_folder, filename)

                # Load the data
                data = np.load(file_path)

                # Check for zero presence
                has_zero_img1, has_zero_img2 = check_images_for_zeros(data)
                if not has_zero_img1 and not has_zero_img2:
                    # If no zeros, copy the file to the destination
                    shutil.copy(file_path, os.path.join(dest_folder, filename))
                    valid_files += 1
                else:
                    # Log the skipped file
                    log_file.write(f"{filename}\n")
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
    # Example usage
    train_folder_path = '/PATH_2_DATASET/SEN12FLOOD_PAIRS/train'
    test_folder_path = '/PATH_2_DATASET/SEN12FLOOD_PAIRS/test'
    output_folder_path = '/PATH_2_DATASET/SEN12FLOOD_PAIRS_CLEAN'

    create_clean_dataset(train_folder_path, test_folder_path, output_folder_path)
