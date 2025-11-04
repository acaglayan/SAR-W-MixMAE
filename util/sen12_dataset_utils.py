# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import csv
import rasterio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def center_crop(image, target_size=(512, 512)):
    _, h, w = image.shape
    top = (h - target_size[0]) // 2
    left = (w - target_size[1]) // 2
    return image[:, top:top + target_size[0], left:left + target_size[1]]


def custom_sort(key):
    if isinstance(key, tuple) and len(key) > 0:
        # Try to convert the first element of the tuple to an integer
        try:
            return 0, int(key[0])  # Sort by the integer value of the first element
        except ValueError:
            return 1, key  # If conversion fails, sort it as a string
    return 2, key  # For other types, sort last


def load_and_process_image(file_path):
    try:
        # image = Image.open(file_path)
        # image_data = np.array(image, dtype=np.float32)  # Convert to float32
        # Open the GeoTIFF file with Rasterio
        with rasterio.open(file_path) as src:
            image = src.read(1).astype(np.float32)  # Read the first band and cast to float32

        # Replace invalid values (e.g., NaNs or Infs)
        if not np.isfinite(image).all():
            image_data = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        return image
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Error loading image at {file_path}: {str(e)}")


class SEN12FloodDataset(Dataset):
    def __init__(self, dataset_root, json_file, split_csv, transform=None):
        self.dataset_root = dataset_root
        self.json_file = json_file
        self.split_csv = split_csv
        self.transform = transform

        # Load folder split information from split_csv
        self.folder_list = self._load_split()

        # Load the metadata from JSON
        with open(self.json_file, 'r') as f:
            self.data_metadata = json.load(f)

        # Extract only relevant folders from metadata based on split
        self.samples = {folder: self.data_metadata[folder] for folder in self.folder_list if
                        folder in self.data_metadata}

    def _load_split(self):
        # Load the train/test split from CSV file
        folder_list = []
        with open(self.split_csv, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                folder_list.append(row[0])
        return folder_list

    def __len__(self):
        return len(self.samples)

    def create_pairs(self, saving_dir, log_file_path):
        missing_files_log = []

        # Iterate through the folders and create image pairs
        for folder_name, folder_data in self.samples.items():
            # Separate images based on flooding labels
            flooding_false_entries = []
            other_entries = []

            # Iterate through each sub-dictionary in the folder data
            for key, entry in sorted(folder_data.items(), key=custom_sort):
                if key.isdigit():  # Ensure we are processing the numbered sub-dictionaries
                    entry_data = {
                        'filename': entry['filename'],
                        'flooding': entry['FLOODING'],
                        'filepath_vh': os.path.join(self.dataset_root, folder_name,
                                                    entry['filename'] + '_corrected_VH.tif'),
                        'filepath_vv': os.path.join(self.dataset_root, folder_name,
                                                    entry['filename'] + '_corrected_VV.tif')
                    }
                    if not entry['FLOODING']:
                        flooding_false_entries.append(entry_data)
                    other_entries.append(entry_data)

            # Create pairs of data ensuring no duplicates or reverse order
            processed_pairs = set()  # Track pairs that have already been processed
            # Create pairs of data ensuring non-redundant combinations
            for false_entry in flooding_false_entries:
                for other_entry in other_entries:
                    # Skip if the entries are the same
                    if false_entry == other_entry:
                        continue

                    # Sort the entries to always have the smaller key first
                    entry_pair = tuple(sorted([false_entry['filename'], other_entry['filename']]))
                    # Check if this pair has already been processed
                    if entry_pair in processed_pairs:
                        continue

                    # Mark this pair as processed
                    processed_pairs.add(entry_pair)
                    try:
                        # Load both images (VH and VV channels for each entry)
                        vh_1 = load_and_process_image(false_entry['filepath_vh'])
                        vv_1 = load_and_process_image(false_entry['filepath_vv'])
                        vh_2 = load_and_process_image(other_entry['filepath_vh'])
                        vv_2 = load_and_process_image(other_entry['filepath_vv'])

                        # Stack VH and VV for each image
                        image_1 = np.stack([vh_1, vv_1], axis=0)
                        image_2 = np.stack([vh_2, vv_2], axis=0)

                        if (image_1.shape[1] != 512 or image_1.shape[2] != 512 or
                                image_2.shape[1] != 512 or image_2.shape[2] != 512):
                            # Log the skipped pair due to smaller dimensions
                            missing_files_log.append(
                                f"Skipping pair due to smaller dimensions: {false_entry['filename']} and "
                                f"{other_entry['filename']}")
                            continue
                        # Apply center crop
                        # retain the main content of the image and discard the edges
                        # image_1 = center_crop(image_1)
                        # image_2 = center_crop(image_2)

                        # Stack image pairs (2x2x512x512)
                        image_pair = np.stack([image_1, image_2], axis=0)

                        # Determine the label for the image pair
                        label = 0 if not other_entry['flooding'] else 1  # 0: No Flooding, 1: Flooding

                        # Save the image pair and label
                        pair_name = f"{folder_name}__{false_entry['filename']}__{other_entry['filename']}"
                        pair_file_path = os.path.join(saving_dir, f"{pair_name}__{label}.npy")
                        np.save(pair_file_path, image_pair)
                    except FileNotFoundError as e:
                        missing_files_log.append(str(e))
                        continue
                    except Exception as e:
                        missing_files_log.append(
                            f"Unexpected error for pair {false_entry['filename']} and {other_entry['filename']}: {str(e)}")
                        continue

        # Log missing files
        with open(log_file_path, 'w') as log_file:
            for log_entry in missing_files_log:
                log_file.write(log_entry + '\n')


def calculate_mean_std(data_root):
    vh_values = []
    vv_values = []

    # Iterate through all `.npy` files in the dataset
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                image_pair = np.load(file_path)
                # Extract VH and VV channels from both images in the pair
                vh_values.append(image_pair[0, 0, :, :].flatten())
                vv_values.append(image_pair[0, 1, :, :].flatten())
                vh_values.append(image_pair[1, 0, :, :].flatten())
                vv_values.append(image_pair[1, 1, :, :].flatten())

    # Concatenate all values for each channel
    vh_values = np.concatenate(vh_values)
    vv_values = np.concatenate(vv_values)

    # Calculate mean and std for each channel
    mean_vh = np.mean(vh_values)
    std_vh = np.std(vh_values)
    mean_vv = np.mean(vv_values)
    std_vv = np.std(vv_values)

    return [mean_vh, mean_vv], [std_vh, std_vv]


def calculate_mean_std_streaming(data_root):
    vh_means = []
    vv_means = []
    vh_vars = []
    vv_vars = []
    total_pixels = 0

    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                image_pair = np.load(file_path).astype(np.float32)

                # Compute mean for each channel (VH and VV)
                vh_mean = np.mean(image_pair[:, 0, :, :])
                vv_mean = np.mean(image_pair[:, 1, :, :])
                vh_means.append(vh_mean)
                vv_means.append(vv_mean)

                # Compute variance for each channel
                vh_var = np.var(image_pair[:, 0, :, :])
                vv_var = np.var(image_pair[:, 1, :, :])
                vh_vars.append(vh_var)
                vv_vars.append(vv_var)

    # Calculate dataset-wide mean and std
    mean_vh = np.mean(vh_means)
    mean_vv = np.mean(vv_means)
    std_vh = np.sqrt(np.mean(vh_vars))
    std_vv = np.sqrt(np.mean(vv_vars))

    return [mean_vh, mean_vv], [std_vh, std_vv]


def visualize_image_pair(pair_file_path):
    # Load the saved image pair
    image_pair = np.load(pair_file_path)

    # Extract VV and VH for both images in the pair
    image_1 = image_pair[0]  # First image in the pair
    image_2 = image_pair[1]  # Second image in the pair

    vh_1, vv_1 = image_1[0], image_1[1]  # First image: VH and VV channels
    vh_2, vv_2 = image_2[0], image_2[1]  # Second image: VH and VV channels

    # Convert images to numpy arrays and apply normalization
    vh_1 = normalize_image(vh_1)
    vv_1 = normalize_image(vv_1)
    vh_2 = normalize_image(vh_2)
    vv_2 = normalize_image(vv_2)

    # Plotting VV and VH channels for the first and second images side by side
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # First image
    axes[0, 0].imshow(vh_1, cmap='gray')
    axes[0, 0].set_title('Image 1 - VH')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vv_1, cmap='gray')
    axes[0, 1].set_title('Image 1 - VV')
    axes[0, 1].axis('off')

    # Second image
    axes[1, 0].imshow(vh_2, cmap='gray')
    axes[1, 0].set_title('Image 2 - VH')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(vv_2, cmap='gray')
    axes[1, 1].set_title('Image 2 - VV')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def normalize_image(image, lower_percentile=2, upper_percentile=98):
    # Replace NaN and Inf values with appropriate values
    image = np.nan_to_num(image, nan=0.0, posinf=np.max(image[np.isfinite(image)]),
                          neginf=np.min(image[np.isfinite(image)]))

    # Compute lower and upper percentile for normalization
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)

    # Clip values and normalize
    if lower_bound == upper_bound:
        return np.zeros_like(image)

    image = np.clip(image, lower_bound, upper_bound)
    image = (image - lower_bound) / (upper_bound - lower_bound + 1e-5)  # Add epsilon to avoid division by zero
    return image


def visualize_tif_image(folder_path, filename):
    # Define paths for VV and VH corrected files
    vh_file_path = os.path.join(folder_path, filename + '_corrected_VH.tif')
    vv_file_path = os.path.join(folder_path, filename + '_corrected_VV.tif')

    try:
        # Load VH and VV channels using rasterio
        with rasterio.open(vh_file_path) as vh_src:
            vh_array = vh_src.read(1)  # Read the first band

        with rasterio.open(vv_file_path) as vv_src:
            vv_array = vv_src.read(1)  # Read the first band

        # Convert images to numpy arrays and apply normalization
        vh_normalized = normalize_image(vh_array)
        vv_normalized = normalize_image(vv_array)

        # Plotting VV and VH channels side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(vh_normalized, cmap='gray')
        axes[0].set_title('VH Channel (Normalized)')
        axes[0].axis('off')

        axes[1].imshow(vv_normalized, cmap='gray')
        axes[1].set_title('VV Channel (Normalized)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def count_labels(folder_path):
    true_count = 0
    false_count = 0

    # Iterate through all the files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a .npy file
        if file_name.endswith(".npy"):
            # Extract the label from the filename, which is the last part before '.npy'
            label = file_name.split("__")[-1].replace(".npy", "")

            if label == "1":
                true_count += 1
            elif label == "0":
                false_count += 1

    return true_count, false_count


def main_create_dataset():
    # Define paths
    dataset_root = r"PATH_2_DATASET\SEN12FLOOD_ALIGNED"
    saving_dir_train = r"PATH_2_DATASET\SEN12FLOOD_PAIRS\train"
    saving_dir_test = r"PATH_2_DATASET\SEN12FLOOD_PAIRS\test"
    json_file = os.path.join(dataset_root, "S1list.json")
    train_csv = "train.csv"
    test_csv = "test.csv"
    train_log_file = os.path.join(saving_dir_train, "train_log.txt")
    test_log_file = os.path.join(saving_dir_test, "test_log.txt")

    # Create output directories if they don't exist
    os.makedirs(saving_dir_train, exist_ok=True)
    os.makedirs(saving_dir_test, exist_ok=True)

    # Create the dataset and data loaders
    train_dataset = SEN12FloodDataset(dataset_root, json_file, train_csv)
    test_dataset = SEN12FloodDataset(dataset_root, json_file, test_csv)

    # Create pairs for training/test dataset
    train_dataset.create_pairs(saving_dir_train, train_log_file)
    test_dataset.create_pairs(saving_dir_test, test_log_file)

    # Print summary
    print("Training data pairs saved.")
    print("Test data pairs saved.")


def main_mean_std():
    # Calculate mean and std for the entire dataset
    dataset_root = r"PATH_2_DATASET\SEN12FLOOD_PAIRS"
    mean, std = calculate_mean_std_streaming(dataset_root)
    print(f"Mean: {mean}, Std: {std}")
    # OLD: Mean: [np.float32(0.027356267), np.float32(0.14240828)], Std: [np.float32(0.17907965), np.float32(1.4645792)]
    # NEW: Mean: [np.float32(0.027194295), np.float32(0.14155816)], Std: [np.float32(0.17784561), np.float32(1.4349885)]


def main_visuzalize():
    pair_file_path = r"PATH_2_DATASET\SEN12FLOOD_PAIRS\train\0140__S1A_IW_GRDH_1SDV_20190218T030905_20190218T030930_025978_02E4EB_98B8__S1B_IW_GRDH_1SDV_20190401T030823_20190401T030856_015607_01D42A_6D09__1.npy"  # Replace with your actual path to a saved pair
    visualize_image_pair(pair_file_path)

    # Example Usage
    folder_path = r"PATH_2_DATASET\SEN12FLOOD\0332"  # Replace with your actual folder path
    filename = 'S1A_IW_GRDH_1SDV_20190118T194433_20190118T194458_025536_02D511_B847'  # Replace with the actual file name without channel suffix
    visualize_tif_image(folder_path, filename)


def main_stats():
    # Define paths to your training and testing folders
    train_folder = r"PATH_2_DATASET\SEN12FLOOD_PAIRS\train"
    test_folder = r"PATH_2_DATASET\SEN12FLOOD_PAIRS\test"
    # Count labels for training and testing sets
    train_true_count, train_false_count = count_labels(train_folder)
    test_true_count, test_false_count = count_labels(test_folder)

    # Print out the counts
    print("Training Set Counts:")
    print(f"True (Flooding) Labels: {train_true_count}")
    print(f"False (No Flooding) Labels: {train_false_count}")
    print(f"Total: {train_true_count + train_false_count}\n")

    print("Testing Set Counts:")
    print(f"True (Flooding) Labels: {test_true_count}")
    print(f"False (No Flooding) Labels: {test_false_count}")
    print(f"Total: {test_true_count + test_false_count}")


def main():
    # main_create_dataset()
    # main_mean_std()
    main_visuzalize()
    main_stats()


if __name__ == "__main__":
    main()
