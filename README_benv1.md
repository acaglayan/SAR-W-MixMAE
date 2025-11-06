# BENv1 Migration Guide (Zarr)

This guide explains **how to switch the repository from BENv2 to BENv1**.  
BENv1 here is assumed to be stored as a **Zarr** dataset and split by CSV lists.

> TL;DR — change imports, channel stats, dataset constructors, and point your scripts to the **.zarr** path.

---

## 0) What’s different vs BENv2?

- **Storage**: BENv1 uses a **Zarr** store (`/path/to/benv1.zarr`) instead of BENv2’s folder layout.
- **Loader**: Use `sarwmix.bigearthnetv1.BigEarthNetv1` (not `BigEarthNetv2`).
- **Channel statistics (dB)**:  
  - BENv1 **mean** = `[-19.2309, -12.5951]`  
  - BENv1 **std**  = `[  3.1672,   2.9644]`
- **CSV splits**: Place your split files in `datasets/benv1_train.csv`, `datasets/benv1_val.csv`, `datasets/benv1_test.csv`.
- **CSV format**: The loader expects a CSV where the **S1 patch identifier is in column index 1** by default (`id_col=1`). If your CSV differs, pass `id_col=<col_idx>` to the dataset or adjust your CSVs.
- **Labels mapping**: The loader reads `datasets/label_indices.json` (BigEarthNet-19 mapping). Ensure that file exists.

---

## 1) Data Preparation Checklist

- ✅ A Zarr store at **`/path/to/benv1.zarr`** with structure:  
  `sentinel1/<patch_id>/VH` and `sentinel1/<patch_id>/VV` arrays.
- ✅ CSVs with the **S1 patch id in column 1** (2nd column), e.g.
  ```csv
  idx, S1_patch_id
  0, S1A_IW_GRDH_1SDV_20190201T051200_20190201T051225_025703_02E2F1_58
  ...
  ```
  If your patch id is in a different column, set `id_col` accordingly.
- ✅ `datasets/label_indices.json` present with the BE19 mapping.
- ✅ BENv1 stats (dB) ready: `mean=[-19.2309, -12.5951]`, `std=[3.1672, 2.9644]`.

---

## 2) Code edits — Pretraining (`main_pretrain.py`)

**Replace the import**:
```diff
- from sarwmix.bigearthnetv2 import BigEarthNetv2
+ from sarwmix.bigearthnetv1 import BigEarthNetv1
```

**Replace channel stats and normalization** (keep stats in **dB**, matching the loader output):
```diff
- benv2_mean = [-19.1966, -12.5144]
- benv2_std  = [  5.4501,   5.0019]
- normalize = transforms.Normalize(mean=benv2_mean, std=benv2_std)
+ benv1_mean = [-19.2309, -12.5951]
+ benv1_std  = [  3.1672,   2.9644]
+ normalize = transforms.Normalize(mean=benv1_mean, std=benv1_std)
```

**Switch the dataset constructor**:
```diff
- dataset_train = BigEarthNetv2(
-     csv_path='datasets/benv2_train_val.csv',
-     root_dir=args.data_path,
-     transform=transform_train
- )
+ dataset_train = BigEarthNetv1(
+     zarr_path=args.data_path,
+     csv_paths=('datasets/benv1_train.csv', 'datasets/benv1_val.csv'),
+     transform=transform_train
+ )
```

> Notes:
> - `args.data_path` should now be the **path to your BENv1 `.zarr` store**.
> - The loader returns tuples like `(sar_tensor, labels, weights)`; pretraining usually uses `sar_tensor` and (optionally) `weights` for reconstruction loss. Keep your linear-vs-dB logic consistent: **inputs to encoder in dB**, **weights computed from linear** (handled in your utilities).

---

## 3) Code edits — Fine-tuning (`main_finetune.py`)

**Replace the import**:
```diff
- from sarwmix.bigearthnetv2 import BigEarthNetv2
+ from sarwmix.bigearthnetv1 import BigEarthNetv1
```

**Replace channel stats and normalization**:
```diff
- benv2_mean = [-19.1966, -12.5144]
- benv2_std  = [  5.4501,   5.0019]
- normalize_train = transforms.Normalize(mean=benv2_mean, std=benv2_std)
- normalize_val   = transforms.Normalize(mean=benv2_mean, std=benv2_std)
- normalize_test  = transforms.Normalize(mean=benv2_mean, std=benv2_std)
+ benv1_mean = [-19.2309, -12.5951]
+ benv1_std  = [  3.1672,   2.9644]
+ normalize_train = transforms.Normalize(mean=benv1_mean, std=benv1_std)
+ normalize_val   = transforms.Normalize(mean=benv1_mean, std=benv1_std)
+ normalize_test  = transforms.Normalize(mean=benv1_mean, std=benv1_std)
```

**Switch the dataset constructors**:
```diff
- dataset_train = BigEarthNetv2(csv_path='datasets/benv2_train.csv', root_dir=args.data_path, transform=transform_train)
- dataset_val   = BigEarthNetv2(csv_path='datasets/benv2_val.csv',   root_dir=args.data_path, transform=transform_val)
- dataset_test  = BigEarthNetv2(csv_path='datasets/benv2_test.csv',  root_dir=args.data_path, transform=transform_test)
+ dataset_train = BigEarthNetv1(zarr_path=args.data_path, csv_paths=['datasets/benv1_train.csv'], transform=transform_train)
+ dataset_val   = BigEarthNetv1(zarr_path=args.data_path, csv_paths=['datasets/benv1_val.csv'],   transform=transform_val)
+ dataset_test  = BigEarthNetv1(zarr_path=args.data_path, csv_paths=['datasets/benv1_test.csv'],  transform=transform_test)
```

> Tip: Update any help text for `--data_path` to read “**path to BENv1 `.zarr`**”.

---

## 4) Scripts — Point to BENv1 paths

In both your **ABCI** and **local** scripts, update these two lines:

```bash
MODELs_PATH="/<PATH_FOR_SAVING_MODELS_BENv1>"
DATA_PATH="/<PATH_TO_BENv1_DATASET>.zarr"
```

- Use a **separate output directory** for BENv1 (to avoid mixing with BENv2 checkpoints).  
- Keep the rest of the script identical (launcher is still `mpirun`).

---

## 5) Quick sanity checks

1) **Import smoke test**
```bash
python - <<'PY'
from sarwmix.bigearthnetv1 import BigEarthNetv1
print("OK: imports")
PY
```

2) **One-batch dataloader check** (replace paths as needed)
```bash
python - <<'PY'
from sarwmix.bigearthnetv1 import BigEarthNetv1
from torchvision import transforms
from torch.utils.data import DataLoader

mean=[-19.2309,-12.5951]; std=[3.1672,2.9644]
tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
ds = BigEarthNetv1(zarr_path="/path/to/benv1.zarr",
                   csv_paths=["datasets/benv1_train.csv"],
                   transform=tr)
x, y, w = ds[0]
print("sample shapes:", x.shape, y.shape, w.shape)  # expect (2,128,128), (19,), (128,128) or per-token weights
PY
```

3) **Mini pretrain run (local 2 GPUs)**
```bash
TASK_TYPE=pretrain MODEL=base EPOCH=64 ./scripts/local_mpi.sh
```

4) **Mini finetune run (local 2 GPUs)**
```bash
TASK_TYPE=finetune MODEL=base EPOCH=64 ./scripts/local_mpi.sh
```

> If your CSV’s S1 patch id isn’t in column **1**, pass `id_col=<col_idx>` to `BigEarthNetv1(...)` or fix the CSVs.

---

## 6) Notes on domains

- Keep the **encoder inputs in dB** but compute the **polarization-aware weights in linear** (your utilities already follow this).  
- BENv1 and BENv2 label spaces are both mapped to **BigEarthNet-19** in this repo setup; ensure `datasets/label_indices.json` matches your splits.

---

## 7) Reverting back to BENv2

- Revert the import lines, stats, and dataset constructors in `main_pretrain.py` and `main_finetune.py` to the BENv2 versions shown in the original README.
- Point `DATA_PATH` back to the BENv2 root and switch your CSV names to `benv2_*`.

---

## 8) FAQ

- **Q:** My CSVs only contain one column with the patch id.  
  **A:** That’s fine — either put the id in **column 1** (second column) or pass `id_col=0` to the dataset.

- **Q:** I get shape mismatches in transforms.  
  **A:** Ensure your transform produces `(2,128,128)` tensors and that the normalization stats are BENv1’s dB values above.

- **Q:** Where does the label mapping come from?  
  **A:** `datasets/label_indices.json` (BigEarthNet-19). If you move it, adjust the dataset’s `label_json` argument.
