# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import os
import re
import csv, json
from collections import OrderedDict
from typing import Optional, Sequence, Tuple
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import dataset, DataLoader
import zarr
from tqdm import tqdm
from weighting import compute_weights


class BigEarthNetv1(dataset.Dataset):
    def __init__(
        self,
        zarr_path: str,
        csv_path: Sequence[str],
        transform=None,
        weight_mode: Optional[str] = "per_channel",  # 'per_channel' | 'shared' | None
        exp_scale: float = 1.0,
        label_json: str = "datasets/label_indices.json",
        id_col: int = 1,  # S1 id column index in CSV
    ):
        super().__init__()
        self.z = zarr.open(zarr_path, mode="r")
        self.s1 = self.z["sentinel1"]
        self.bands = ("VH", "VV")
        self.transform = transform
        self.weight_mode = weight_mode
        self.exp_scale = exp_scale
        self.id_col = id_col
        
        # resolve the label_json path dynamically relative to the project root
        project_root = Path(__file__).resolve().parent.parent  # goes up two levels from the current file location
        label_json_path = project_root / label_json  # construct the absolute path to the label JSON
        
         # Check if the file exists
        if not label_json_path.exists():
            raise FileNotFoundError(f"The file {label_json_path} does not exist.")

        # collect sample ids from CSVs
        self.samples = []
        for p in csv_path:
            with open(p, "r", newline="") as f:
                r = csv.reader(f)
                for row in r:
                    if not row:
                        continue
                    self.samples.append(row[self.id_col])

        with open(label_json_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        self.original_labels = label_data["original_labels"]           # str -> int (original idx)
        self.label_conversion = label_data["label_conversion"]         # list[list[int]] original->BE19 groups
        self.num_classes = len(label_data["BigEarthNet-19_labels"])
        # original idx -> BE19 idx
        self.orig_to_be19 = {}
        for be_idx, orig_list in enumerate(self.label_conversion):
            for o in orig_list:
                self.orig_to_be19[o] = be_idx

    def __len__(self) -> int:
        return len(self.samples)

    def _read_s1(self, patch_id: str) -> torch.Tensor:
        # zarr arrays are stored as (H,W); convert to torch (2,H,W)
        g = self.s1[patch_id]
        vh = torch.tensor(g["VH"][:], dtype=torch.float32)
        vv = torch.tensor(g["VV"][:], dtype=torch.float32)
        img = torch.stack([vh, vv], dim=0)
        return img

    def _labels_from_attrs(self, patch_id: str) -> torch.Tensor:
        g = self.s1[patch_id]
        raw_labels = list(g.attrs["labels"])  # list of strings
        multi_hot = torch.zeros(self.num_classes, dtype=torch.int64)
        for lab in raw_labels:
            # map string -> original idx
            if lab not in self.original_labels:
                continue
            orig_idx = self.original_labels[lab]
            if orig_idx in self.orig_to_be19:
                be_idx = self.orig_to_be19[orig_idx]
                multi_hot[be_idx] = 1
        return multi_hot

    def __getitem__(self, idx: int):
        pid = self.samples[idx]
        sar_data = self._read_s1(pid)  # (2,H,W) in dB

        if self.transform is None:
            raise ValueError("Please provide a torchvision transform (e.g., resize to 128x128).")
        sar_data = self.transform(sar_data)  # important: weights must follow the same resizing

        weights = compute_weights(sar_data, mode=self.weight_mode)

        labels = self._labels_from_attrs(pid)  # (19,)
        return sar_data, labels, weights



