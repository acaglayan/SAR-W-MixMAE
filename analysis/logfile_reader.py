# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import os
import re
import pandas as pd

# Set display options to show all columns
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame from being truncated


def extract_epoch_number(line):
    # Regular expression to match the finetune path and extract the epoch number
    # match = re.search(r"output_dir='.*?/ckpt_ft_(\d+)'", line)
    match = re.search(r"output_dir='.*?/base_(\d+)'", line)  # revision code
    if match:
        # Extract and return the epoch number
        return int(match.group(1))
    else:
        # match = re.search(r"output_dir='.*?ckpt_ft_(\d+_\S+)'", line)
        match = re.search(r"output_dir='.*?base_(\d+_\S+)'", line)   # revision
        if match:
            return match.group(1)

    return None


def parse_metrics_from_file(file_path):
    """
    Parse the metrics and best epoch from a given log file.
    """
    metrics = {
        'Macro_Avg_Precision': None,
        'Micro_Avg_Precision': None,
        'Macro_F1': None,
        'Micro_F1': None,
        'Macro_Precision': None,
        'Micro_Precision': None
    }
    best_epoch = None
    epoch_number = None
    capture_metrics = False  # Flag to track when to capture the metrics

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Check if the line contains the epoch number
            if "output_dir" in line:
                epoch_number = extract_epoch_number(line)
            # Check for the best epoch information
            if "Best Results Summary: Epoch" in line:
                best_epoch = int(re.findall(r'\d+', line)[-1])
                capture_metrics = True  # Start capturing the following metric table
            # Only capture the specific metrics table after "Best Results Summary"
            if capture_metrics:
                # Check for metrics in the table
                if "| Average Precision " in line:
                    metrics['Macro_Avg_Precision'] = float(line.split('|')[2].strip())
                    metrics['Micro_Avg_Precision'] = float(line.split('|')[3].strip())
                elif "| F1 Score " in line:
                    metrics['Macro_F1'] = float(line.split('|')[2].strip())
                    metrics['Micro_F1'] = float(line.split('|')[3].strip())
                elif "| Precision " in line:
                    metrics['Macro_Precision'] = float(line.split('|')[2].strip())
                    metrics['Micro_Precision'] = float(line.split('|')[3].strip())

                # Once the table is fully captured, stop capturing to avoid unnecessary data
                if metrics['Macro_Precision'] and metrics['Micro_Precision']:
                    capture_metrics = False

    return epoch_number, best_epoch, metrics


def parse_all_files(folder_path):
    """
    Parse all matching files in a folder to extract metrics data.
    """
    # Define regex pattern for matching filenames
    # pattern = re.compile(r'.*finetune\.sh\.o(\d+)')
    pattern = re.compile(r'^.*pretrain_base-(\d+)-base_') # revision regex

    metrics_data = {
        'Epoch': [],
        'Best_Epoch': [],
        'Macro_Avg_Precision': [],
        'Micro_Avg_Precision': [],
        'Macro_F1': [],
        'Micro_F1': [],
        # 'Macro_Precision': [],
        # 'Micro_Precision': []
    }

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            file_path = os.path.join(folder_path, filename)
            # Extract the best epoch and metrics from the file
            epoch_number, best_epoch, metrics = parse_metrics_from_file(file_path)
            if best_epoch is not None:
                # Append the epoch and metrics to the metrics_data dictionary
                metrics_data['Epoch'].append(epoch_number)
                metrics_data['Best_Epoch'].append(best_epoch)
                metrics_data['Macro_Avg_Precision'].append(metrics['Macro_Avg_Precision'])
                metrics_data['Micro_Avg_Precision'].append(metrics['Micro_Avg_Precision'])
                metrics_data['Macro_F1'].append(metrics['Macro_F1'])
                metrics_data['Micro_F1'].append(metrics['Micro_F1'])
                # metrics_data['Macro_Precision'].append(metrics['Macro_Precision'])
                # metrics_data['Micro_Precision'].append(metrics['Micro_Precision'])

    return metrics_data


# Example usage:
folder_path = r'<PATH-2->\logs'  # Change this to your actual folder path
metrics_data = parse_all_files(folder_path)

# Convert metrics_data to a DataFrame for better viewing
df_metrics = pd.DataFrame(metrics_data)
print(df_metrics.sort_values(['Macro_F1', 'Macro_Avg_Precision'],
                             ascending=[False, False]))
