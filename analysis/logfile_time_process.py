# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import os
import re
from datetime import datetime, timedelta


def extract_time_from_line(line):
    """Extracts training time from 'Training time hh:mm:ss' format."""
    match = re.search(r"Training time (\d+):(\d+):(\d+)", line)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    return None


def extract_timestamps_from_file(file_path):
    """Extracts first and last timestamp from a log file."""
    timestamp_pattern = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d+)\]")
    first_timestamp, last_timestamp = None, None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = timestamp_pattern.search(line)
            if match:
                time_str = match.group(1)
                if first_timestamp is None:
                    first_timestamp = time_str  # first occurrence
                last_timestamp = time_str  # continuously update last occurrence

    if first_timestamp and last_timestamp:
        start_time = datetime.strptime(first_timestamp, "%H:%M:%S.%f")
        end_time = datetime.strptime(last_timestamp, "%H:%M:%S.%f")
        return end_time - start_time
    return None


def process_logs(root_dir, output_log_file):
    log_data = []
    total_process_time = timedelta()
    total_files = 0

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".log") and file != "run_output.log":
                log_path = os.path.join(root, file)
                total_files += 1
                process_time = None

                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        process_time = extract_time_from_line(line)
                        if process_time:  # Found "Training time hh:mm:ss"
                            break

                if not process_time:  # If no "Training time" line, check timestamps
                    process_time = extract_timestamps_from_file(log_path)

                if process_time:
                    log_data.append(f"{file} {process_time}")
                    total_process_time += process_time

    # Write results to output log file
    with open(output_log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_data))
        f.write(f"\nTotal file count: {total_files} Total process time: {total_process_time}\n")


# Example usage
root_directory = r"<PATH-2->\logs\SAR-W-MixMAE"  # Change this to the actual logs directory
output_log = r"<PATH-2->\logs\SAR-W-MixMAE\output_log.txt"  # Change this to the desired output file

process_logs(root_directory, output_log)
