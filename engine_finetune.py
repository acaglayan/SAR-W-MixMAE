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
from tabulate import tabulate
import torch

from torchmetrics.classification import MultilabelAveragePrecision  
from torchmetrics.classification import MultilabelF1Score  
from torchmetrics.classification import MultilabelPrecision  
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

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda'):
            outputs = model(samples)
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
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()

    # define torchmetrics objects
    macro_avr_precision = MultilabelAveragePrecision(num_labels=19, average="macro")
    micro_avr_precision = MultilabelAveragePrecision(num_labels=19, average="micro")
    macro_f1_score = MultilabelF1Score(num_labels=19, average="macro")
    micro_f1_score = MultilabelF1Score(num_labels=19, average="micro")
    macro_precision = MultilabelPrecision(num_labels=19, average="macro")
    micro_precision = MultilabelPrecision(num_labels=19, average="micro")

    macro_avr_precision.to(device)
    micro_avr_precision.to(device)
    macro_f1_score.to(device)
    micro_f1_score.to(device)
    macro_precision.to(device)
    micro_precision.to(device)

    # MetricLogger for monitoring loss, etc.
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Switch model to eval mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        targets = batch[1]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Sigmoid for multi-label probabilities
        probabilities = torch.sigmoid(outputs)

        # Update each metric object with the batch predictions
        macro_avr_precision.update(probabilities, targets)
        micro_avr_precision.update(probabilities, targets)
        macro_f1_score.update(probabilities, targets)
        micro_f1_score.update(probabilities, targets)
        macro_precision.update(probabilities, targets)
        micro_precision.update(probabilities, targets)

        # Log the loss value for this batch
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # Now compute the final dataset-level metrics
    avr_prc_macro_val = macro_avr_precision.compute().item()
    avr_prc_micro_val = micro_avr_precision.compute().item()
    f1_macro_val = macro_f1_score.compute().item()
    f1_micro_val = micro_f1_score.compute().item()
    prc_macro_val = macro_precision.compute().item()
    prc_micro_val = micro_precision.compute().item()
    
    # Store the final values in metric_logger
    if 'avr_prc_macro' not in metric_logger.meters:
        metric_logger.add_meter('avr_prc_macro', misc.SmoothedValue(window_size=1))
    if 'avr_prc_micro' not in metric_logger.meters:
        metric_logger.add_meter('avr_prc_micro', misc.SmoothedValue(window_size=1))
    if 'prc_macro' not in metric_logger.meters:
        metric_logger.add_meter('prc_macro', misc.SmoothedValue(window_size=1))
    if 'prc_micro' not in metric_logger.meters:
        metric_logger.add_meter('prc_micro', misc.SmoothedValue(window_size=1))
    if 'f1_macro' not in metric_logger.meters:
        metric_logger.add_meter('f1_macro', misc.SmoothedValue(window_size=1))
    if 'f1_micro' not in metric_logger.meters:
        metric_logger.add_meter('f1_micro', misc.SmoothedValue(window_size=1))

    metric_logger.meters['avr_prc_macro'].update(avr_prc_macro_val, n=1)
    metric_logger.meters['avr_prc_micro'].update(avr_prc_micro_val, n=1)
    metric_logger.meters['prc_macro'].update(prc_macro_val, n=1)
    metric_logger.meters['prc_micro'].update(prc_micro_val, n=1)
    metric_logger.meters['f1_macro'].update(f1_macro_val, n=1)
    metric_logger.meters['f1_micro'].update(f1_micro_val, n=1)

    # pack the metric results (Macro, Micro) in a dict
    results = {
        "Average Precision": [metric_logger.avr_prc_macro.global_avg, metric_logger.avr_prc_micro.global_avg],
        "F1 Score": [metric_logger.f1_macro.global_avg, metric_logger.f1_micro.global_avg],
        "Precision": [metric_logger.prc_macro.global_avg, metric_logger.prc_micro.global_avg]
    }

    # create a table with tabulate
    table = tabulate(
        [[metric, f"{macro:.4f}", f"{micro:.4f}"] for metric, (macro, micro) in results.items()],
        headers=["Metric", "Macro", "Micro"],
        tablefmt="pipe"
    )
    print("\n" + table)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
