# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import os

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from osgeo import gdal


def align_sentinel1_to_sentinel2(sentinel1_path, sentinel2_path, output_path, success_log, error_log):
    """
    Align Sentinel-1 GeoTIFF to match the spatial reference and extent of a Sentinel-2 GeoTIFF.

    Parameters:
    sentinel1_path (str): Path to Sentinel-1 GeoTIFF file.
    sentinel2_path (str): Path to Sentinel-2 GeoTIFF file (reference for projection).
    output_path (str): Path to save the aligned Sentinel-1 output.
    success_log (file object): Log file to log successful operations.
    error_log (file object): Log file to log errors.
    """
    try:
        # Open the Sentinel-2 reference file to get its projection and geotransform
        with gdal.Open(sentinel2_path) as s2_dataset:
            if not s2_dataset:
                raise FileNotFoundError(f"Sentinel-2 file not found: {sentinel2_path}")

            # Get the projection and geotransform from Sentinel-2
            projection = s2_dataset.GetProjection()
            geotransform = s2_dataset.GetGeoTransform()

            # Perform the reprojection and alignment of Sentinel-1 to match Sentinel-2
            options = gdal.WarpOptions(
                format='GTiff',
                xRes=geotransform[1],  # Resolution in X
                yRes=-geotransform[5],  # Resolution in Y
                outputBounds=[geotransform[0],
                              geotransform[3] + geotransform[5] * s2_dataset.RasterYSize,
                              geotransform[0] + geotransform[1] * s2_dataset.RasterXSize,
                              geotransform[3]],  # Set the output bounds
                dstSRS=projection  # Set the spatial reference to match Sentinel-2
            )

            # Reproject using gdal.Warp
            gdal.Warp(output_path, sentinel1_path, options=options)

            # Log the successful operation
            success_log.write(f"Aligned: {sentinel1_path} -> {output_path}\n")
    except Exception as e:
        # Log the error
        error_log.write(f"Error aligning {sentinel1_path}: {e}\n")


def main():
    # Define paths
    dataset_root = r"\PATH_2_DATASET\SEN12FLOOD"
    aligned_dataset_root = r"\PATH_2_DATASET\SEN12FLOOD_ALIGNED"

    # Define log file paths
    success_log_path = os.path.join(aligned_dataset_root, "alignment_success.log")
    error_log_path = os.path.join(aligned_dataset_root, "alignment_errors.log")

    # Open log files for writing
    with open(success_log_path, 'w') as success_log, open(error_log_path, 'w') as error_log:

        # Iterate over all folders in the dataset
        for folder_name in os.listdir(dataset_root):
            folder_path = os.path.join(dataset_root, folder_name)

            # Skip if not a directory
            if not os.path.isdir(folder_path):
                continue

            # Create corresponding folder in the aligned dataset root
            aligned_folder_path = os.path.join(aligned_dataset_root, folder_name)
            os.makedirs(aligned_folder_path, exist_ok=True)

            # Locate Sentinel-1 files and Sentinel-2 reference file (e.g., B04)
            s1_files = [f for f in os.listdir(folder_path) if f.startswith("S1") and f.endswith(".tif")]
            s2_files = [f for f in os.listdir(folder_path) if f.startswith("S2") and f.endswith("B04.tif")]

            if not s2_files:
                error_log.write(f"No Sentinel-2 B04 file found in folder: {folder_name}\n")
                continue

            # Choose the first Sentinel-2 file for alignment (since all represent the same area)
            sentinel2_path = os.path.join(folder_path, s2_files[0])

            # Align each Sentinel-1 file to match the chosen Sentinel-2 file
            for s1_file in s1_files:
                sentinel1_path = os.path.join(folder_path, s1_file)
                output_path = os.path.join(aligned_folder_path, s1_file)

                # Align Sentinel-1 to Sentinel-2
                align_sentinel1_to_sentinel2(sentinel1_path, sentinel2_path, output_path, success_log, error_log)


def visualize_s1_s2(dataset_root, aligned_dataset_root, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each folder in the dataset
    for folder_name in os.listdir(dataset_root):
        folder_path_s2 = os.path.join(dataset_root, folder_name)
        folder_path_s1 = os.path.join(aligned_dataset_root, folder_name)

        # Skip if not a directory
        if not os.path.isdir(folder_path_s2) or not os.path.isdir(folder_path_s1):
            continue

        s2_rgb_files = None

        # Locate Sentinel-2 files and group them by date
        for file_name in os.listdir(folder_path_s2):
            if file_name.startswith("S2") and file_name.endswith(".tif"):
                # Extract the date part from the filename (assuming it follows the pattern S2_YYYY-MM-DD_BXX.tif)
                date_str = file_name.split('_')[1]
                # Check if the RGB bands (B02, B03, B04) are available for the current date
                b02_file = os.path.join(folder_path_s2, f"S2_{date_str}_B02.tif")
                b03_file = os.path.join(folder_path_s2, f"S2_{date_str}_B03.tif")
                b04_file = os.path.join(folder_path_s2, f"S2_{date_str}_B04.tif")

                if os.path.exists(b02_file) and os.path.exists(b03_file) and os.path.exists(b04_file):
                    s2_rgb_files = {
                        "B02": b02_file,
                        "B03": b03_file,
                        "B04": b04_file
                    }
                    break

        # Locate a VH file for Sentinel-1 in the aligned folder
        s1_vh_file = None
        for file_name in os.listdir(folder_path_s1):
            if file_name.startswith("S1") and "VH" in file_name and file_name.endswith(".tif"):
                s1_vh_file = os.path.join(folder_path_s1, file_name)
                break

        # Ensure all necessary files are available
        if not s1_vh_file or not s2_rgb_files:
            print(f"Missing files in folder: {folder_name}, skipping...")
            continue

        try:
            # Read Sentinel-1 VH file
            with rasterio.open(s1_vh_file) as s1_src:
                s1_vh_data = s1_src.read(1)

            # Read Sentinel-2 RGB bands and stack them into an RGB image
            with rasterio.open(s2_rgb_files["B02"]) as s2_b02_src, \
                 rasterio.open(s2_rgb_files["B03"]) as s2_b03_src, \
                 rasterio.open(s2_rgb_files["B04"]) as s2_b04_src:

                b02 = s2_b02_src.read(1)  # Blue
                b03 = s2_b03_src.read(1)  # Green
                b04 = s2_b04_src.read(1)  # Red

                # Stack the bands to create an RGB image (normalize between 0-1 for visualization)
                s2_rgb = np.stack([b04, b03, b02], axis=-1)
                s2_rgb = s2_rgb.astype(np.float32)
                s2_rgb /= np.max(s2_rgb)  # Normalize for visualization

            # Plot side-by-side visualization of Sentinel-1 VH and Sentinel-2 RGB
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot Sentinel-1 VH
            ax1.imshow(s1_vh_data, cmap='gray')
            ax1.set_title(f"Sentinel-1 VH - {folder_name}")
            ax1.axis('off')

            # Plot Sentinel-2 RGB
            ax2.imshow(s2_rgb)
            ax2.set_title(f"Sentinel-2 RGB - {folder_name}")
            ax2.axis('off')

            # Save the visualization
            output_file = os.path.join(output_dir, f"{folder_name}_comparison.png")
            plt.tight_layout()
            plt.savefig(output_file, dpi=150)
            plt.close(fig)

            print(f"Saved visualization for folder: {folder_name}")

        except Exception as e:
            print(f"Error processing folder {folder_name}: {e}")
            continue

    print("All visualizations saved.")


def main_visualize():
    # Define paths
    dataset_root = r"\PATH_2_DATASET\SEN12FLOOD"  # Original Sentinel-2 data
    aligned_dataset_root = r"\PATH_2_DATASET\SEN12FLOOD_ALIGNED"  # Aligned Sentinel-1 data
    output_dir = r"\PATH_2_DATASET\SEN12FLOOD_VISUALIZATIONS"  # Directory to save visualizations

    # Run the visualization function
    visualize_s1_s2(dataset_root, aligned_dataset_root, output_dir)


if __name__ == "__main__":
    main_visualize()
