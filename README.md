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
    sen12_align_s1_to_s2.py, sen12_data_prep.py, sen12_prune_partial_pairs.py,
    sen12flood_loader.py, weighting.py

  scripts/                  # training/eval scripts for local environment and ABCI server environment, SEN12FLOOD dataset preparation script (MIT)
  datasets/                 # CSVs/splits as needed (MIT)
  LICENSES/                 # MIT.txt, NOASSERTION.txt
  NOTICE                    # provenance + permission note
  THIRD_PARTY.md            # file-by-file mapping table
  CITATION.cff
  requirements.txt
  INSTALL.md
  README_benv1.md
  README_sen12.md
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
- **Migration guide:** see [README_benv1.md](README_benv1.md).

### SEN12-FLOOD (fine-tuning only)

- **what**: binary flood detection on pairwise SAR (VV/VH) inputs. each sample has two timepoints  
`(img1 = non-flood, img2 = flood | non-flood)`, shaped `(2 × 2 × 512 × 512)`. default `--patch_op avg`.

- **data prep**: run `scripts/prepare_sen12flood.sh` (zip or raw-root). it performs:
1) unzip (if zip given) → 2) align S1→S2 grid → 3) build pairs → 4) clean partial-coverage pairs.  
outputs: `CURATED_SEN12FLOOD/{train,test}` and prints class counts. requires **gdal**.  
full guide: [`README_sen12.md`](./README_sen12.md).

- **normalization (dB, keep BENv1 stats)**  
`mean = [-19.2309, -12.5951]`  
`std  = [  3.1672,   2.9644]`

- **checkpoints**: finetune from **BENv1** pretraining, e.g.  
`--finetune $MODELs_PATH/PRETr_CKPTs_LOGs/benv1_rand_pretrain_base/checkpoint_64.pth`
- **See full migration guide:** at [README_sen12.md](README_sen12.md).
---

## Training
**Launcher:** `mpirun` (OpenMPI) + NCCL backend. See the sections below for the exact commands we use on ABCI and on a local 2‑GPU machine. For all options, run:
```bash
python main_pretrain.py -h
python main_finetune.py -h
```

---
### Self-supervised pretraining
**Hardware / node**
- **Cluster:** ABCI (single node)
- **GPUs:** 8 × NVIDIA **H200** (script also handles V100=4 GPUs, A100=8)
- **CPU:** `#PBS -l select=1:ncpus=192`

**Distributed launch**
- We use **MPI** (`mpirun`) to spawn one Python process per GPU and **NCCL** for the backend.
- GPU count and hosts are inferred from `nvidia-smi` and `$PBS_NODEFILE`.

**Batch / schedule**
- **Per-GPU batch:** `256` → **Global batch:** `256 × 8 = 2048`
- Planned **1024** epochs with **40** warmup; we **cut at epoch 64** (intentional cut) and released that checkpoint (`checkpoint_64.pth`). The finetuning and evaluation below use `checkpoint_64.pth`.


> **Note**: The polarization‑aware weighting for reconstruction is used during pretraining.

### Finetuning
Per-GPU batch `128` (global `128 × 8 GPUs = 1024`), 50 epochs

## Checkpoints
Release artifacts (pretrained weights and fine‑tuned heads) will be added as GitHub Releases.
TODO: After uploading, update this section with links and example `--finetune` usage.

---

## Citation
If you find this work useful, please cite the paper (see [`CITATION.cff`](`CITATION.cff`)).

---

## License & Provenance
- **Original files** in `sarwmix/`, `scripts/`, `datasets/` → **MIT** (see [`LICENSES/MIT.txt`](LICENSES/MIT.txt)).  
- **Upstream MixMIM/MixMAE files** (verbatim or modified) in `util/` and selected top‑level `main_*.py`, `engine_*.py`, `models_*.py` → **SPDX: NOASSERTION**, redistributed with **written permission (2025‑10‑15)**. See `NOTICE` and `THIRD_PARTY.md`.

See [`NOTICE`](NOTICE) for the permission note and [`THIRD_PARTY.md`](THIRD_PARTY.md) for the file‑by‑file mapping.

## References (background only):
MAE (He et al.), MixMAE/MixMIM (Li et al.), Swin Transformer (Liu et al.), BEiT (Bao et al.). We cite these as prior work; our code is based only MixMIM here, under explicit permission. We credit all these works and thank the authors.
