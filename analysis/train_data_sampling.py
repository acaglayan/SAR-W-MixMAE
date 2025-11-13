# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Define the data
data_percentage = [100.0, 50.0, 25.0, 12.5]
pretrained_macro_avg_precision = [0.7141, 0.6928, 0.6616, 0.6453]
pretrained_micro_avg_precision = [0.8391, 0.8245, 0.8139, 0.8046]
pretrained_macro_f1 = [0.6443, 0.6293, 0.6016, 0.5894]
pretrained_micro_f1 = [0.7384, 0.7261, 0.7190, 0.7076]

random_macro_avg_precision = [0.6107, 0.5781, 0.5545, 0.5268]
random_micro_avg_precision = [0.7713, 0.7339, 0.7139, 0.6827]
random_macro_f1 = [0.4936, 0.4713, 0.4295, 0.4058]
random_micro_f1 = [0.6750, 0.6509, 0.6233, 0.5977]

# Define a colorblind-friendly palette and markers
cb_palette = {
    "Macro Average Precision": "black",
    "Micro Average Precision": "gray",
    "Macro F1": "darkblue",
    "Micro F1": "darkgreen",
}
markers = {
    "Macro Average Precision": "o",  # Circle
    "Micro Average Precision": "s",  # Square
    "Macro F1": "^",         # Triangle Up
    "Micro F1": "D",         # Diamond
}
line_styles = {
    "Pretrained": "-",
    "Random": "--",
}

# Create the figure and axes for side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True, dpi=200)

# Pretrained Model Plot
axes[0].plot(data_percentage, pretrained_macro_avg_precision, marker=markers["Macro Average Precision"],
             label="Macro AP", linestyle=line_styles["Pretrained"], color=cb_palette["Macro Average "
                                                                                                    "Precision"])
axes[0].plot(data_percentage, pretrained_micro_avg_precision, marker=markers["Micro Average Precision"],
             label="Micro AP", linestyle=line_styles["Pretrained"], color=cb_palette["Micro Average "
                                                                                                    "Precision"])
axes[0].plot(data_percentage, pretrained_macro_f1, marker=markers["Macro F1"],
             label="Macro F1", linestyle=line_styles["Pretrained"], color=cb_palette["Macro F1"])
axes[0].plot(data_percentage, pretrained_micro_f1, marker=markers["Micro F1"],
             label="Micro F1", linestyle=line_styles["Pretrained"], color=cb_palette["Micro F1"])
axes[0].set_xlabel("Percentage of Training Data")
axes[0].set_ylabel("Performance")
axes[0].set_title("Pretrained Model")
axes[0].legend()
axes[0].grid(True)

# Random Initialization Model Plot
axes[1].plot(data_percentage, random_macro_avg_precision, marker=markers["Macro Average Precision"],
             label="Macro AP", linestyle=line_styles["Random"], color=cb_palette["Macro Average Precision"])
axes[1].plot(data_percentage, random_micro_avg_precision, marker=markers["Micro Average Precision"],
             label="Micro AP", linestyle=line_styles["Random"], color=cb_palette["Micro Average Precision"])
axes[1].plot(data_percentage, random_macro_f1, marker=markers["Macro F1"],
             label="Macro F1", linestyle=line_styles["Random"], color=cb_palette["Macro F1"])
axes[1].plot(data_percentage, random_micro_f1, marker=markers["Micro F1"],
             label="Micro F1", linestyle=line_styles["Random"], color=cb_palette["Micro F1"])
axes[1].set_xlabel("Percentage of Training Data")
axes[1].set_title("Random Initialization Model")
axes[1].legend(loc="upper left")
axes[1].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig("data_reduction_performance.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()
