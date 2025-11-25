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

import argparse
import datetime
import json
import numpy as np
import os
import io
import time
from pathlib import Path
from tabulate import tabulate

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from torch.nn import MultiLabelSoftMarginLoss 
from torchvision.transforms import transforms

from sarwmix.bigearthnetv2 import BigEarthNetv2

import util.lr_decay as lrd
import torch.distributed as dist
import util.misc as misc
import sarwmix.helper as helper
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# import models_vit
from timm.models import create_model

from engine_finetune import train_one_epoch, evaluate
import models_mixmim_ft


def get_args_parser():
    parser = argparse.ArgumentParser('MixMIM fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')

    # Model parameters
    parser.add_argument('--model', default='mixmim_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=128, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='./output_dir/checkpoint.pth',
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=19, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./ckpt_ft',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log_ft',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--port', default=29530, type=int)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    bigearthnetv2_mean = [-19.1966, -12.5144]
    bigearthnetv2_std = [5.4501, 5.0019]

    transform_train = transforms.Compose([
        transforms.Resize(size=(args.input_size, args.input_size)),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=bigearthnetv2_mean, std=bigearthnetv2_std)])

    transform_val = transforms.Compose([
        transforms.Resize(size=(args.input_size, args.input_size)),  # 3 is bicubic
        transforms.Normalize(mean=bigearthnetv2_mean, std=bigearthnetv2_std)])
        
    transform_test = transforms.Compose([
        transforms.Resize(size=(args.input_size, args.input_size)),
        transforms.Normalize(mean=bigearthnetv2_mean, std=bigearthnetv2_std)
    ])

    dataset_train = BigEarthNetv2(csv_path='datasets/benv2_train.csv', root_dir=args.data_path, transform=transform_train)
    dataset_val = BigEarthNetv2(csv_path='datasets/benv2_val.csv', root_dir=args.data_path, transform=transform_val)                        
    dataset_test = BigEarthNetv2(csv_path='datasets/benv2_test.csv', root_dir=args.data_path, transform=transform_test)

    global_rank = misc.get_rank()

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
                
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    if args.finetune and not args.eval:
        if args.finetune.startswith('s3:'):
            from petrel_client.client import Client
            client = Client()
            with io.BytesIO(client.get(args.finetune)) as f:
                checkpoint = torch.load(f, map_location='cpu', weights_only=False)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    if global_rank == 0:
        model.eval()
        in_chans = 2
        flops = FlopCountAnalysis(model, torch.rand(1, in_chans, args.input_size, args.input_size).to(device))
        print(flop_count_table(flops, max_depth=2))
        model.train()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = MultiLabelSoftMarginLoss() 

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        val_stats = evaluate(data_loader_val, model, device)
        print(f"Average precision macro of the network on the {len(dataset_val)} val images: {val_stats['avr_prc_macro']:.3f}%")

        test_stats = evaluate(data_loader_test, model, device)
        print(f"Average precision macro of the network on the {len(dataset_test)} test images: {test_stats['avr_prc_macro']:.3f}%")
        
        # clean shutdown of distributed
        if args.distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    use_ceph = args.resume.startswith('s3:')
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_avg_prc_macro = 0.0
    best_score = float('-inf')
    best_default_score = float('-inf')
    best_stats = {}
    best_default_stats = {}
    best_epoch = 0
    best_default_epoch = 0
    # most best result approach
    best_stats_mbr = None
    best_epoch_mbr = None

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

        test_stats = evaluate(data_loader_val, model, device)
        
        # Save the best model
        current_score = (test_stats['f1_macro'] * 1.3) + (test_stats['f1_micro'] * 1.1) + (test_stats['avr_prc_macro'] * 1.2) + (test_stats['prc_macro'] * 1.2)
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_stats = {
                "avr_prc_macro": test_stats['avr_prc_macro'],
                "avr_prc_micro": test_stats['avr_prc_micro'],
                "f1_macro": test_stats['f1_macro'],
                "f1_micro": test_stats['f1_micro'],
                "prc_macro": test_stats['prc_macro'],
                "prc_micro": test_stats['prc_micro']
            }
            
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
        
        # ============ Default BEST RESULTS code ===========
        default_score = test_stats['f1_macro'] + test_stats['f1_micro'] + test_stats['avr_prc_macro'] + test_stats['avr_prc_micro'] + test_stats['prc_macro'] + test_stats['prc_micro']
        if default_score > best_default_score:
            best_default_score = default_score
            best_default_epoch = epoch
            best_default_stats = {
                "avr_prc_macro": test_stats['avr_prc_macro'],
                "avr_prc_micro": test_stats['avr_prc_micro'],
                "f1_macro": test_stats['f1_macro'],
                "f1_micro": test_stats['f1_micro'],
                "prc_macro": test_stats['prc_macro'],
                "prc_micro": test_stats['prc_micro']
            }
            
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best_default.pth')
            misc.save_on_master(to_save, checkpoint_path)
            print(f"Best [default] model saved at epoch {epoch} with score {best_default_score:.4f}")
        
        # ============ Default BEST RESULTS code ===========        
        
        # ============ MOST BEST RESULTS code ===========
        # Build a dict of the 6 metrics for the new epoch
        new_mbr_dict = {
            'ap_macro':  test_stats['avr_prc_macro'],
            'ap_micro':  test_stats['avr_prc_micro'],
            'f1_macro':  test_stats['f1_macro'],
            'f1_micro':  test_stats['f1_micro'],
            'prec_macro': test_stats['prc_macro'],
            'prec_micro': test_stats['prc_micro'],
        }

        if best_stats_mbr is None:
            # first epoch => automatically best
            best_stats_mbr = new_mbr_dict
            best_epoch_mbr = epoch

            # save a separate checkpoint for the 'most best results' approach
            checkpoint_path_mbr = os.path.join(args.output_dir, 'checkpoint_best_mbr.pth')
            to_save_mbr = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            misc.save_on_master(to_save_mbr, checkpoint_path_mbr)
            print(f"MBR: Best model (first) at epoch {epoch}")
        else:
            # compare with current best
            if helper.is_new_better_mbr(new_mbr_dict, best_stats_mbr):
                best_stats_mbr = new_mbr_dict
                best_epoch_mbr = epoch
                # save checkpoint to keep the best by MBR approach
                checkpoint_path_mbr = os.path.join(args.output_dir, 'checkpoint_best_mbr.pth')
                to_save_mbr = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }
                misc.save_on_master(to_save_mbr, checkpoint_path_mbr)
                print(f"MBR: Best model updated at epoch {epoch}")

        # ============ MOST BEST RESULTS code ===========
        
        print(f"Average precision macro and micro of the network on the {len(dataset_val)} test images: {test_stats['avr_prc_macro']:.2f}% and {test_stats['avr_prc_micro']:.2f}%")
        max_avg_prc_macro = max(max_avg_prc_macro, test_stats["avr_prc_macro"])
        print(f'Max Average precision - macro: {max_avg_prc_macro:.2f}%')

        if log_writer is not None:
            # log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch) # commented on 2024-07-31
            # log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch) # commented on 2024-07-31
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

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
    
    # print the best metrics in a formatted table
    results = {
        "Metric": ["Average Precision", "F1 Score", "Precision"],
        "Macro": [best_stats['avr_prc_macro'], best_stats['f1_macro'], best_stats['prc_macro']],
        "Micro": [best_stats['avr_prc_micro'], best_stats['f1_micro'], best_stats['prc_micro']]
    }

    table = tabulate(
        [[metric, f"{macro:.4f}", f"{micro:.4f}"] for metric, macro, micro in zip(results["Metric"], results["Macro"], results["Micro"])],
        headers=["Metric", "Macro", "Micro"],
        tablefmt="pipe"
    )
    print("Best Results Summary: Epoch {}".format(best_epoch))
    print("\n" + table)
    
    # clean shutdown of distributed training
    if args.distributed and dist.is_initialized():
        # make sure all ranks finished logging / saving
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
