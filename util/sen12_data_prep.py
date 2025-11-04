# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

from torch.utils.data import Dataset
from osgeo import gdal, osr
import os
import json
import csv
import numpy as np
import rasterio
from torchvision.transforms import transforms


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


def reproject_sentinel1_to_sentinel2(s1_file, s2_reference_file, output_path):
    """
    Reproject and align Sentinel-1 data to match Sentinel-2 reference using GDAL.

    Parameters:
    s1_file: str - The path to the Sentinel-1 GeoTIFF file to be aligned.
    s2_reference_file: str - The path to the reference Sentinel-2 GeoTIFF file.
    output_path: str - The path where the reprojected file will be saved.
    """
    # Open the Sentinel-2 reference to get the spatial reference and geographic boundaries
    s2_dataset = gdal.Open(s2_reference_file)
    if not s2_dataset:
        raise RuntimeError(f"Unable to open Sentinel-2 reference file: {s2_reference_file}")

    # Get the target coordinate system and geographic boundaries from Sentinel-2
    s2_proj = s2_dataset.GetProjection()
    s2_geo_transform = s2_dataset.GetGeoTransform()
    s2_x_size = s2_dataset.RasterXSize
    s2_y_size = s2_dataset.RasterYSize

    # Define the output resolution to match Sentinel-2
    x_res = s2_geo_transform[1]
    y_res = s2_geo_transform[5]

    # Run gdalwarp to reproject Sentinel-1 to Sentinel-2's CRS and extent
    warp_options = gdal.WarpOptions(
        format='GTiff',
        srcSRS=s1_file,
        dstSRS=s2_proj,
        xRes=x_res,
        yRes=-y_res,
        width=s2_x_size,
        height=s2_y_size,
        outputBounds=(s2_geo_transform[0], s2_geo_transform[3],
                      s2_geo_transform[0] + s2_geo_transform[1] * s2_x_size,
                      s2_geo_transform[3] + s2_geo_transform[5] * s2_y_size)
    )

    # Perform the re-projection and save the output
    gdal.Warp(output_path, s1_file, options=warp_options)

    # Clean up
    s2_dataset = None


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
                    s1_entry_data = {
                        'filename': entry['filename'],
                        'flooding': entry['FLOODING'],
                        'filepath_vh': os.path.join(self.dataset_root, folder_name,
                                                    entry['filename'] + '_corrected_VH.tif'),
                        'filepath_vv': os.path.join(self.dataset_root, folder_name,
                                                    entry['filename'] + '_corrected_VV.tif')
                    }
                    # Get a Sentinel-2 reference file (e.g., S2_B02, S2_B03, S2_B04)
                    s2_reference_file = os.path.join(self.dataset_root, folder_name,
                                                     "S2_2019-01-24_B04.tif")  # Use B04 as an RGB reference
                    # Align Sentinel-1 to Sentinel-2 before further processing
                    try:
                        vh_reprojected_path = s1_entry_data['filepath_vh'].replace('.tif', '_aligned.tif')
                        vv_reprojected_path = s1_entry_data['filepath_vv'].replace('.tif', '_aligned.tif')

                        if not os.path.exists(vh_reprojected_path):
                            reproject_sentinel1_to_sentinel2(
                                s1_file=s1_entry_data['filepath_vh'],
                                s2_reference_file=s2_reference_file,
                                output_path=vh_reprojected_path
                            )
                        if not os.path.exists(vv_reprojected_path):
                            reproject_sentinel1_to_sentinel2(
                                s1_file=s1_entry_data['filepath_vv'],
                                s2_reference_file=s2_reference_file,
                                output_path=vv_reprojected_path
                            )

                        # Update file paths after alignment
                        s1_entry_data['filepath_vh'] = vh_reprojected_path
                        s1_entry_data['filepath_vv'] = vv_reprojected_path

                    except RuntimeError as e:
                        missing_files_log.append(str(e))
                        continue

                    # Append entries based on flooding label
                    if not entry['FLOODING']:
                        flooding_false_entries.append(s1_entry_data)
                    other_entries.append(s1_entry_data)

            # Create pairs of data ensuring non-redundant combinations
            processed_pairs = set()  # Track pairs that have already been processed
            for false_entry in flooding_false_entries:
                for other_entry in other_entries:
                    # Skip if the entries are the same
                    if false_entry == other_entry:
                        continue

                    # Sort the entries to always have the smaller key first
                    entry_pair = tuple(sorted([false_entry['filename'], other_entry['filename']]))
                    if entry_pair in processed_pairs:
                        continue
                    processed_pairs.add(entry_pair)

                    try:
                        # Load and stack the aligned images
                        vh_1 = load_and_process_image(false_entry['filepath_vh'])
                        vv_1 = load_and_process_image(false_entry['filepath_vv'])
                        vh_2 = load_and_process_image(other_entry['filepath_vh'])
                        vv_2 = load_and_process_image(other_entry['filepath_vv'])

                        image_1 = np.stack([vh_1, vv_1], axis=0)
                        image_2 = np.stack([vh_2, vv_2], axis=0)

                        # Apply center crop
                        image_1 = center_crop(image_1)
                        image_2 = center_crop(image_2)

                        # Stack image pairs
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


def main_create_dataset():
    # Define paths
    dataset_root = r"\PATH_2_DATASET\SEN12FLOOD"
    saving_dir_train = r"\PATH_2_DATASET\SEN12FLOOD_PAIRS\train"
    saving_dir_test = r"\PATH_2_DATASET\SEN12FLOOD_PAIRS\test"
    json_file = os.path.join(dataset_root, "S1list.json")
    train_csv = "train.csv"
    test_csv = "test.csv"
    train_log_file = os.path.join(saving_dir_train, "train_log.txt")
    test_log_file = os.path.join(saving_dir_test, "test_log.txt")

    # Create output directories if they don't exist
    os.makedirs(saving_dir_train, exist_ok=True)
    os.makedirs(saving_dir_test, exist_ok=True)

    # Define transformations (if needed)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create the dataset and data loaders
    train_dataset = SEN12FloodDataset(dataset_root, json_file, train_csv, transform=transform)
    test_dataset = SEN12FloodDataset(dataset_root, json_file, test_csv, transform=transform)

    # Create pairs for training/test dataset
    train_dataset.create_pairs(saving_dir_train, train_log_file)
    test_dataset.create_pairs(saving_dir_test, test_log_file)

    # Print summary
    print("Training data pairs saved.")
    print("Test data pairs saved.")


def main():
    main_create_dataset()


if __name__ == "__main__":
    main()
