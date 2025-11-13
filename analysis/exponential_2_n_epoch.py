# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd

from analysis.logfile_reader import parse_all_files

# ---- read + build dataframe ----
folder_path = r'<PATH-2->\logs'
log = parse_all_files(folder_path)

df = pd.DataFrame({
    'Epoch': log['Epoch'],
    'Macro_Avg_Precision': log['Macro_Avg_Precision'],
    'Micro_Avg_Precision': log['Micro_Avg_Precision'],
    'Macro_F1': log['Macro_F1'],
    'Micro_F1': log['Micro_F1'],
})

# ---- expected epochs, dedup (mean), sort ----
expected_epochs = [1, 2, 4, 8, 16, 32, 40, 64, 128, 256, 512, 600, 1024]
df['Epoch'] = pd.to_numeric(df['Epoch'], errors='coerce')
df = (df.dropna(subset=['Epoch'])
        .loc[lambda x: x['Epoch'].isin(expected_epochs)]
        .groupby('Epoch', as_index=False).mean()
        .sort_values('Epoch').reset_index(drop=True))

# ---- styling (consistent with your other figure) ----
cb_palette = {
    "Macro Average Precision": "black",
    "Micro Average Precision": "gray",
    "Macro F1": "darkblue",
    "Micro F1": "darkgreen",
}
markers = {
    "Macro Average Precision": "o",   # circle
    "Micro Average Precision": "s",   # square
    "Macro F1": "^",                  # triangle up
    "Micro F1": "D",                  # diamond
}
# Linestyles: solid for AP, dashed for F1
ls = {"AP": "-", "F1": "--"}

# ---- one-panel plot with four lines + vertical reference lines ----
fig, ax = plt.subplots(figsize=(18, 6), dpi=200)

# metric curves
ax.plot(df['Epoch'], df['Macro_Avg_Precision'],
        marker=markers["Macro Average Precision"], linestyle=ls["AP"],
        color=cb_palette["Macro Average Precision"], label="Macro AP")
ax.plot(df['Epoch'], df['Micro_Avg_Precision'],
        marker=markers["Micro Average Precision"], linestyle=ls["AP"],
        color=cb_palette["Micro Average Precision"], label="Micro AP")
ax.plot(df['Epoch'], df['Macro_F1'],
        marker=markers["Macro F1"], linestyle=ls["F1"],
        color=cb_palette["Macro F1"], label="Macro F1")
ax.plot(df['Epoch'], df['Micro_F1'],
        marker=markers["Micro F1"], linestyle=ls["F1"],
        color=cb_palette["Micro F1"], label="Micro F1")

# vertical lines: powers of two (orange, faint), plus 40 (green) and 600 (red)
powers_of_two = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
special_epochs = {40: "green", 600: "red"}

# one legend entry for all 2^n lines
for i, e in enumerate(powers_of_two):
    if e == 64:
        # highlight epoch 64: solid + thicker
        ax.axvline(e, linestyle='--', linewidth=2.5, color='orange', alpha=0.9, zorder=1)
    else:
        ax.axvline(e, linestyle='--', color='orange', alpha=0.5, zorder=0,
                   label=r'Epoch $2^n$' if i == 0 else None)

# separate legend entries for 40 and 600
for e, c in special_epochs.items():
    ax.axvline(e, linestyle='--', color=c, zorder=0, label=f'Epoch {e}')

# axes, labels, legend
ax.set_title('Average Precision (AP) & F1 by Epoch (with reference epochs marked)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Performance')
ax.set_xticks(expected_epochs)
ax.set_xticklabels([str(e) for e in expected_epochs])
ax.grid(True, axis='y', which='major')   # Only horizontal gridlines (remove vertical x-gridlines)

# dedupe legend entries (since axvline loops can create dupes)
handles, labels = ax.get_legend_handles_labels()
seen = set()
uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
ax.legend(*zip(*uniq), ncol=2)

# Optional: log-2 x-axis for nicer spacing
# ax.set_xscale('log', base=2)

# choose which epochs to label on the x-axis
hide = {2, 4, 8, 32}
ticks_to_show = [e for e in expected_epochs if e not in hide]  # -> [1,16,40,64,128,256,512,600,1024]
ax.set_xticks(ticks_to_show)
ax.set_xticklabels([str(e) for e in ticks_to_show])
ax.tick_params(axis='x', rotation=0)

plt.tight_layout()

# Optional save (before show)
plt.savefig("exponential_performance.png", dpi=300, bbox_inches="tight", pad_inches=0.05)

plt.show()
