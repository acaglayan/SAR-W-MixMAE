# SAR-W-MixMAE: Polarization‑Aware Self‑Supervised Pretraining for Masked Autoencoders on SAR Data

This repository contains the code for the paper **“SAR-W-MixMAE: Polarization‑Aware Self‑Supervised Pretraining for Masked Autoencoders on SAR Data.”**  
It builds on MixMIM/MixMAE with a Swin backbone and introduces **per‑pixel polarization‑aware weighting** in the reconstruction loss.

**Key idea — per‑channel weighting**: VH and VV are normalized **in linear scale**, transformed via `exp(1 − norm)`, aggregated to token weights, and applied to per‑token MSE. Inputs to the encoder are in **dB**, while weighting is computed in **linear**, ensuring domain consistency.

## Highlights
- Swin + MixMIM/MixMAE pretraining with mask ratio `r = 0.5`, input `2×128×128 (VH, VV)`.
- Polarization‑aware pixel weights → aggregated token weights for reconstruction MSE.
- Strong results on **BigEarthNet v1/v2** (multi‑label) and **SEN12‑FLOOD** fine‑tuning.
- Upstream files from MixMIM are redistributed **with written permission (2025‑10‑15)** and tagged `SPDX: NOASSERTION`; all original files here are **MIT**.

## Repository Layout
```
SAR-W-MixMAE/
  main_pretrain.py          # modified from MixMIM (NOASSERTION)
  main_finetune.py          # modified from MixMIM (NOASSERTION)
  engine_pretrain.py        # modified from MixMIM (NOASSERTION)
  engine_finetune.py        # modified from MixMIM (NOASSERTION)
  models_mixmim.py          # modified from MixMIM (NOASSERTION)
  models_mixmim_ft.py       # modified from MixMIM (NOASSERTION)
  models_sen12_ft.py        # new, derived from MixMIM FT (NOASSERTION)

  util/                     # upstream MixMIM files (verbatim or lightly modified) — NOASSERTION
    pos_embed.py (verbatim), lr_sched.py (verbatim), lr_decay.py (verbatim),
    datasets.py (verbatim), crop.py (verbatim), misc.py (modified), README.md

  sarwmix/                  # all original utilities — MIT
    bigearthnetv1.py, bigearthnetv2.py, helper.py,
    sen12_clean_data.py, sen12_data_alignment.py, sen12_data_prep.py,
    sen12_dataset_utils.py, sen12flood_loader.py, weighting.py

  scripts/                  # training/eval scripts for local environment and ABCI server environment (MIT)
  datasets/                 # CSVs/splits as needed (MIT)
  LICENSES/                 # MIT.txt, NOASSERTION.txt
  NOTICE                    # provenance + permission note
  THIRD_PARTY.md            # file-by-file mapping table
  CITATION.cff
  requirements.txt
  INSTALL.md
  README.md
```

---

## Installation (Python 3.12, CUDA 12.x)

We recommend: install PyTorch first (matching your CUDA), then the rest of the Python deps.

```bash
# 1) Create env
conda create -n sarwmix python=3.12 -y
conda activate sarwmix

# 2) Install PyTorch (choose the right CUDA build)
# Example for CUDA 12.x:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3) Install repo dependencies (Torch is intentionally excluded from requirements.txt)
pip install -r requirements.txt
```

> If you build from source or use a system CUDA, verify `torch.cuda.is_available()` and that the CUDA versions match.

---

## Datasets

### BigEarthNet‑v2 (BENv2)
- Input: Sentinel‑1 GRD; we use **VH,VV** channels, **2×128×128** patches.
- You can load BENv2 with the provided dataset class:
  ```python
  from sarwmix.bigearthnetv2 import BigEarthNetv2
  ```
- Expected usage: training splits via CSVs in `datasets/`.

### BigEarthNet‑v1 (BENv1)
- Legacy support; some loaders expect **Zarr** containers.
- Use `sarwmix/bigearthnetv1.py` and train/val/test splits together with labels in `datasets/`.

### SEN12‑FLOOD (fine‑tuning only)
- Finetune with inputs in **dB** (log10), but compute polarization weights in **linear**.
- Use: `sarwmix/sen12flood_loader.py` and helper scripts
  (`sen12_data_alignment.py`, `sen12_data_prep.py`, `sen12_dataset_utils.py`).

**Normalization policy:** Use BEN **dB‑domain** channel stats during finetuning for distribution consistency with pretraining.

---

## Training

> Use `torchrun` for both single‑ and multi‑GPU. Inspect `python main_pretrain.py -h` and `python main_finetune.py -h` for full options.

> **Note**: The polarization‑aware weighting for reconstruction is used during pretraining.

---

## Checkpoints
Release artifacts (pretrained weights and fine‑tuned heads) will be added as GitHub Releases.
TODO: After uploading, update this section with links and example `--finetune` usage.

---

## Citation
If you find this work useful, please cite the paper (see `CITATION.cff`).

---

## License & Provenance
- **Original files** in `sarwmix/`, `scripts/`, `datasets/` → **MIT** (see `LICENSES/MIT.txt`).  
- **Upstream MixMIM/MixMAE files** (verbatim or modified) in `util/` and selected top‑level `main_*.py`, `engine_*.py`, `models_*.py` → **SPDX: NOASSERTION**, redistributed with **written permission (2025‑10‑15)**. See `NOTICE` and `THIRD_PARTY.md`.
- **Upstream:** This repo is based on MixMIM/MixMAE (SenseTime). We did not use code from facebookresearch/mae; MAE is cited as prior work only.

See `NOTICE` for the permission note and `THIRD_PARTY.md` for the file‑by‑file mapping.

## References (background only):
MAE (He et al.), MixMAE/MixMIM (Li et al.), Swin Transformer (Liu et al.), BEiT (Bao et al.). We cite these as prior work; our code is based only MixMIM here, under explicit permission. We credit all these works and thank the authors.
