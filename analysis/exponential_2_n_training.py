# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the provided CSV file
file_path = 'exp_training_loss.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the data
data.head()

# Assume we can calculate the number of steps per epoch from the provided data
# Normalize steps to epochs. Assuming the steps are proportional to epochs
total_epochs = 1024
max_step = data['Step'].max()
data['Epoch'] = data['Step'] / max_step * total_epochs

# Define specific epochs to mark (2^n, 40, 600)
specific_epochs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 40, 600]

# Plot the training loss over epochs
plt.figure(figsize=(16, 4), dpi=200)   # <- bigger canvas + higher resolution
plt.plot(data['Epoch'], data['Value'], label="Training Loss", color='blue')

# Highlight specific epochs
for epoch in specific_epochs:
    if epoch in [40, 600]:
        plt.axvline(x=epoch, linestyle='--', color='green' if epoch == 40 else 'red', label=f"Epoch {epoch}")
    elif epoch == 64:
        # highlight epoch 64: solid + thicker
        plt.axvline(x=epoch, linestyle='--', linewidth=2.5, color='orange', alpha=0.9, zorder=1)
    else:
        plt.axvline(x=epoch, linestyle='--', color='orange', alpha=0.5, label=r'Epoch $2^n$' if epoch == 1 else None)

# >>> Sparse x-axis labels (hide crowded ones 2, 4, 8, 32)
hide = {2, 4, 8, 32}
ticks_to_show = [e for e in sorted(set(specific_epochs)) if e not in hide]
plt.xticks(ticks_to_show, [str(e) for e in ticks_to_show])

# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss by Epoch (with reference epochs marked)')
plt.legend()

# Show the plot
plt.grid(True, axis='y', which='major') # Only horizontal gridlines (remove vertical x-gridlines)
plt.tight_layout()
plt.savefig("training_loss.png",
            dpi=300,                    # final file DPI (can be > figure dpi)
            bbox_inches="tight",
            pad_inches=0.05)
plt.show()
