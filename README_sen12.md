# SEN12FLOOD Finetuning Guide (Flood / No-Flood)

This guide explains how to **prepare the SEN12FLOOD dataset** and **finetune SAR-W-MixMAE** using **BENv1-pretrained** checkpoints. We **do not** pretrain on SEN12FLOOD.

> **TL;DR** — run the prep script to get `CURATED_SEN12FLOOD/{train,test}`, then finetune with `--input_size 512`, keep **BENv1 dB** normalization, and pass `--finetune <BENv1 checkpoint>`.

---

## 0) Contents

- [1) Download](#1-download)
- [2) Extra dependency](#2-extra-dependency)
- [3) Prepare the data (one command)](#3-prepare-the-data-one-command)
- [4) Code changes for finetuning](#4-code-changes-for-finetuning)
  - [4.1 `main_finetune.py` — edits](#41-main_finetunepy--edits)
  - [4.2 `engine_finetune.py` — full file](#42-engine_finetunepy--full-file)
- [5) Running (finetune & test only)](#5-running-finetune--test-only)
- [6) Quick checklist](#6-quick-checklist)
- [7) Notes on pairing & curation](#7-notes-on-pairing--curation)

---

## 1) Download

Dataset (login required):  
<https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection>

Because login is required, `wget` isn’t feasible; download manually.

**Optional: verify the ZIP contents** (example check):

```bash
echo "ZIP File Summary:"
echo "-----------------"

# Total size of the ZIP file (compressed)
zip_size=$(stat -c %s SEN12FLOOD.zip)
echo "File Size: $zip_size bytes"

# Count the total number of files
file_count=$(zipinfo -1 SEN12FLOOD.zip | wc -l)

# Count the number of unique directories (by looking at directory paths)
dir_count=$(zipinfo -1 SEN12FLOOD.zip | awk -F/ '{if(NF>1) print $1"/"$2}' | sort -u | wc -l)

# Output the number of files and directories
echo "Number of Files: $file_count"
echo "Number of Directories: $dir_count"

# Count files by extension (including .directory)
echo "File Types:"
zipinfo -1 SEN12FLOOD.zip | \
awk -F. 'NF>1 {ext=$NF; count[ext]++} END {for (ext in count) print count[ext], "." ext " files"}' | sort -n
```

Typical output we observed:
```
ZIP File Summary:
-----------------
File Size: 13101306693 bytes
Number of Files: 36460
Number of Directories: 341
File Types:
4 .xml files
5 .directory files
6 .json files
54 .png files
36053 .tif files
```

---

## 2) Extra dependency

Data prep uses **GDAL** (Rasterio is already in the project). Install GDAL into your env:

```bash
# conda (recommended)
conda install -c conda-forge gdal
```

---

## 3) Prepare the data (one command)

We provide a single script that does the full pipeline and only keeps the **final curated dataset**.

**SEN12FLOOD end-to-end prep**
1. **unzip** (if a zip is given)  
2. **ALIGN**: S1 → S2 grid (temp)  
3. **PAIR**:  build pair `.npy` (temp)  
4. **CLEAN**: drop partial-coverage pairs (final only)

The script is at: `scripts/prepare_sen12flood.sh`

### Option A — From the ZIP
```bash
./scripts/prepare_sen12flood.sh \
  --zip  /path/to/SEN12FLOOD.zip \
  --work /scratch/sen12_work \
  --out  /data/CURATED_SEN12FLOOD
```

### Option B — From an already unzipped folder
```bash
./scripts/prepare_sen12flood.sh \
  --raw-root /path/to/SEN12FLOOD \
  --work     /scratch/sen12_work \
  --out      /data/CURATED_SEN12FLOOD
```

The script prints stage progress. Final layout:
```
/data/CURATED_SEN12FLOOD/
  train/*.npy
  test/*.npy
```

- Each `.npy` stores a **pair** `(img1, img2)` with shape `(2, 2, 512, 512)` → two timepoints × (VH, VV) × H × W.  
- Intermediates are deleted by default to save disk space.  
- Keep the repo’s `datasets/` files available: `datasets/S1list.json`, `datasets/sen12_train.csv`, `datasets/sen12_test.csv`. The prep script checks for these before running. These are already provided with the zip file under `datasets/`.

**Sanity counts (what we typically see after curation):**
- Train pairs: **11,078**
- Test pairs:  **2,662**
- Label balance is printed at the end by the script (pos/neg).

---

## 4) Code changes for finetuning

We **finetune from BENv1 pretrained checkpoints**. We keep **BENv1 dB normalization** and use a **two-input** head `(img1, img2)` with **CrossEntropyLoss** for binary flood classification.

### 4.1 `main_finetune.py` — edits

**Imports** to include the SEN12FLOOD dataset loader.
```diff
- from sarwmix.bigearthnetv1 import BigEarthNetv1
- import models_mixmim_ft
+ from sarwmix.sen12flood_loader import FloodDataset
+ import models_sen12_ft
```

**Parser additions/changes** to revise the input dimension and how we aggregate two inputs (change detection)
```diff
+ parser.add_argument('--input_size', default=512, type=int)
+ parser.add_argument('--patch_op', default='avg', type=str, metavar='avg',
+                     help='operation used to aggregate patch diffs between (img1, img2)')
```

**Number of classes** We also need to revise the argument for the number of classes to the binary flood detection task. 
```diff
- nb_classes = 19
+ nb_classes = 2  # flood / no-flood
```

**Normalization (keep BENv1, dB domain)**
```diff
- benv2_mean = [-19.1966, -12.5144]
- benv2_std  = [  5.4501,   5.0019]
- normalize_* = transforms.Normalize(mean=benv2_mean, std=benv2_std)
+ benv1_mean = [-19.2309, -12.5951]
+ benv1_std  = [  3.1672,   2.9644]
+ normalize_* = transforms.Normalize(mean=benv1_mean, std=benv1_std)
```

**Datasets (train/test only; remove val)**
```diff
+ path_train = os.path.join(args.data_path, "train")
+ path_test  = os.path.join(args.data_path, "test")
- dataset_train = BigEarthNetv1(zarr_path=args.data_path, csv_path=['datasets/benv1_train.csv'], transform=transform_train)
- dataset_test  = BigEarthNetv1(zarr_path=args.data_path, csv_path=['datasets/benv1_test.csv'],  transform=transform_test)
+ dataset_train = FloodDataset(root_dir=path_train, transform=transform_train)
+ dataset_test  = FloodDataset(root_dir=path_test,  transform=transform_test)
```

**FLOPs probe (two inputs)**
```diff
- in_chans = 2
- flops = FlopCountAnalysis(model, torch.rand(1, in_chans, args.input_size, args.input_size).to(device))
+ in_chans = 2
+ img1 = torch.rand(1, in_chans, args.input_size, args.input_size).to(device)
+ img2 = torch.rand(1, in_chans, args.input_size, args.input_size).to(device)
+ flops = FlopCountAnalysis(model, (img1, img2))
```

**Loss Criterion**
```diff
- criterion = torch.nn.MultiLabelSoftMarginLoss()
+ criterion = torch.nn.CrossEntropyLoss()
```

**Eval + best-ckpt logic**  
Use the **test** split for `--eval` and after each epoch; save best by accuracy as `checkpoint_best.pth`. Keep accuracy / precision / recall / F1 logging.
```python
if args.eval:
        test_stats = evaluate(data_loader_test, model, device, patch_op=args.patch_op)
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc']:.4f}%")
        print(f"Precision: {test_stats['precision']:.4f}")
        print(f"Recall: {test_stats['recall']:.4f}")
        print(f"F1 Score: {test_stats['f1_score']:.4f}")
        exit(0)

    use_ceph = args.resume.startswith('s3:')
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_score = float('-inf')
    best_epoch = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, use_ceph=use_ceph)

        test_stats = evaluate(data_loader_test, model, device, patch_op=args.patch_op)
        
        # Save the best model based on accuracy
        current_score = test_stats['acc']
        
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
                        
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
            misc.save_on_master(to_save, checkpoint_path)
            print(f"Best model saved at epoch {epoch} with score {best_score:.4f}")
                 
        # Print test metrics
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc']:.4f}% ")
        print(f"Precision: {test_stats['precision']:.4f}")
        print(f"Recall: {test_stats['recall']:.4f}")
        print(f"F1 Score: {test_stats['f1_score']:.4f}")
        
        max_accuracy = max(max_accuracy, test_stats["acc"])
        print(f'Max accuracy: {max_accuracy:.4f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc', test_stats['acc'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
            log_writer.add_scalar('perf/test_precision', test_stats['precision'], epoch)
            log_writer.add_scalar('perf/test_recall', test_stats['recall'], epoch)
            log_writer.add_scalar('perf/test_f1', test_stats['f1_score'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    print("Best Results Summary: Epoch {} Best Accuracy Score {}".format(best_epoch, best_score))
```
---

### 4.2 `engine_finetune.py` — full code structure as below

```python
# SPDX-License-Identifier: NOASSERTION
# Copyright (c) SenseTime. All rights reserved.
# Modifications Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# Used and redistributed with permission from the MixMIM authors (2025-10-15).
# See NOTICE for attribution details and LICENSES/ for license texts.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE:  https://github.com/facebookresearch/mae
# MixMIM (upstream): https://github.com/Sense-X/MixMIM
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    patch_op = args.patch_op

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (imgs1, imgs2, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imgs1 = imgs1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            imgs1, targets = mixup_fn(imgs1, targets)
            imgs2, targets = mixup_fn(imgs2, targets)

        with torch.amp.autocast('cuda'):
            outputs = model(imgs1, imgs2, patch_op=patch_op)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), flush=True)
            sys.exit(1)

        loss /= accum_iter
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=False,
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm.item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            if data_iter_step % 50 == 0:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, patch_op='avg'):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_preds = []
    all_targets = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        imgs1 = batch[0]
        imgs2 = batch[1]
        target = batch[-1]
        imgs1 = imgs1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast(device_type='cuda'):
            output = model(imgs1, imgs2, patch_op=patch_op)
            loss = criterion(output, target)

        acc = accuracy(output, target)[0]

        # get the predicted class, the class with the highest score
        _, preds = torch.max(output, 1)
        # collect true labels and predicted labels for precision, recall and F1 score
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.cpu().numpy())

        batch_size = imgs1.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(f'Accuracy {metric_logger.acc.global_avg:.2f} '
          f'Loss {metric_logger.loss.global_avg:.2f} '
          f'Precision {precision:.2f} '
          f'Recall {recall:.2f} '
          f'F1 Score {f1:.2f}')
    # Return a dictionary with all the metrics
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} | {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

---

## 5) Running (finetune & test only)

This task **does not** include pretraining.

- Use **BENv1** checkpoints via `--finetune`, e.g.  
  `--finetune $MODELs_PATH/PRETr_CKPTs_LOGs/benv1_rand_pretrain_base/checkpoint_64.pth`
- Set `DATA_PATH` to the curated dataset root:  
  `DATA_PATH="<PATH_TO>/CURATED_SEN12FLOOD"`
- Your run scripts should expose **two** modes only:
  - **Finetune** → train on `train/`, evaluate on `test/`, save `checkpoint_best.pth`
  - **Test**     → evaluate a chosen checkpoint (e.g., `checkpoint_best.pth`) on `test/`

**Flags to remember (not exhaustive):**
- `--input_size 512`
- `--patch_op avg`
- `--finetune <path-to-BENv1-checkpoint>`
- `--data_path <PATH_TO>/CURATED_SEN12FLOOD`

(Launcher remains `mpirun` as in the repo scripts.)

---

## 6) Quick checklist

- [ ] `CURATED_SEN12FLOOD/{train,test}` exist and contain `.npy` pairs  
- [ ] `--input_size 512` set  
- [ ] BENv1 dB normalization used (`[-19.2309, -12.5951] / [3.1672, 2.9644]`)  
- [ ] Model takes **two inputs** `(img1, img2)` and supports `--patch_op`  
- [ ] `--finetune` points to a valid **BENv1** checkpoint  
- [ ] `checkpoint_best.pth` appears in your output dir after training

---

## 7) Notes on pairing & curation

- Original dataset has **335 sequences** (avg ~14 SAR images, 512×512 @ 10 m).  
- We convert sequences into **pairs**: the first image is **non-flood**; the second is **flood or non-flood**.  
- We drop **partial-coverage** pairs (≥25% zeros in any VV/VH channel at either timepoint).  
- Splits follow the authors’ **train/test** lists; after curation we observed ~**11,078** train pairs and **2,662** test pairs.

---
