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
import io
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from torchvision.utils import save_image
import torchvision.transforms.functional as F

import timm.optim.optim_factory as optim_factory
import torch.distributed as dist
import util.misc as misc
from sarwmix.bigearthnetv2 import BigEarthNetv2
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed

from timm.models import create_model
import timm.models
import models_mixmim

from engine_pretrain import train_one_epoch


# ------------------------------------------------------------
# HELPER FUNCTIONS 
# ------------------------------------------------------------
# Check if epoch is the power of 2 by using bitwise operations
# (epoch & (epoch - 1)) == 0 is to check if a number is a power of 2.
# In binary, powers of 2 have only one 1 bit, so subtracting 1 from such a number, turns all bits below it to 1 and
# flips the 1 bit to 0. The bitwise AND of the number and its predecessor is zero only if the number is a power of 2.
def is_power_of_2(epoch):
    return (epoch > 0) and (epoch & (epoch - 1)) == 0



def build_checkpoint_paths_from_resume(args, epochs_list=None):
    """
    Given args.resume = something like:
        /path/to/checkpoints_folder/checkpoint_256.pth
    We parse the folder path (checkpoints_folder)
    and then build a list of checkpoint file paths for the given epochs_list.

    Example:
        epochs_list = [64, 128, 256, 600, 1024]
        -> [
            /path/to/checkpoints_folder/checkpoint_64.pth,
            /path/to/checkpoints_folder/checkpoint_128.pth,
            ...
           ]
    """
    """if epochs_list is None:
        epochs_list = [64, 128, 256, 600, 1024]

    # Parse out the directory from args.resume
    # e.g. if args.resume is "/some/folder/checkpoint_256.pth"
    # then base_dir = "/some/folder"
    base_dir = os.path.dirname(args.resume)

    # Build the list of checkpoint paths
    checkpoint_paths = [
        os.path.join(base_dir, f"checkpoint_{epoch}.pth") 
        for epoch in epochs_list
    ]"""
    # Define the model types we want to use for paths
    model_types = ["benv2_rand_pretrain_base", "benv2_rand_pretrain_large", "benv2_rand_pretrain_huge"]

    # Parse out the base directory from args.resume
    base_dir = os.path.dirname(args.resume)

    # Extract the path up to the model directory name (ignoring the checkpoint filename)
    # Example: "/path/to/benv2_rand_pretrain_base/checkpoint_256.pth" => "/path/to"
    root_dir = os.path.dirname(base_dir)

    # Build the list of checkpoint paths for the same checkpoint_64.pth in each model directory
    checkpoint_paths = [
        os.path.join(root_dir, model_type, "checkpoint_64.pth") 
        for model_type in model_types
    ]
    return checkpoint_paths


def denormalize(tensor, mean, std):
    """
    Undo normalization for a tensor that was normalized as (x - mean) / std.

    tensor shape: [B, C, H, W]
    mean, std are lists of length = C, e.g. [-19.1966, -12.5144], [5.4501, 5.0019]
    """
    mean_t = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    return tensor * std_t + mean_t


def compare_checkpoints_reconstruction(
    model, 
    checkpoint_paths,           # list of strings (file paths)
    data_loader,
    device,
    output_dir,
    mean, std,
    max_images=5,
    mask_ratio=0.0,
    gap_width=5,                # vertical gap (in pixels) between columns
    upsample_factor=1           # if > 1, upsample final image before saving
):
    """
    1) Loads the same small batch of data (up to max_images) from data_loader.
       (Assumes each batch yields (samples, _, mse_linear_weight).)
    2) For each checkpoint in checkpoint_paths:
       - loads checkpoint
       - does forward pass
       - collects reconstruction
    3) Concatenates [Original, Recon@Epoch1, Recon@Epoch2, ...] side by side,
       inserting a white vertical gap of `gap_width` pixels between each column.
    4) Saves one combined PNG per sample in the batch.

    Arguments:
    ----------
    model : model (same architecture used in training)
    checkpoint_paths : list of strings for .pth checkpoint files
    data_loader : DataLoader that yields (samples, _, mse_linear_weight)
    device : torch device (e.g. torch.device('cuda'))
    output_dir : folder to save the final comparison images
    mean, std : channel-wise normalization stats (lists)
    max_images : how many samples from the batch to visualize
    mask_ratio : for the forward pass partial masking
    gap_width : size (in pixels) of the vertical strip inserted between columns
    upsample_factor : if > 1, enlarge final output by this integer factor 
                      (purely for nicer viewing)

    Returns:
    --------
    None (images are saved to output_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    #####################
    # 1) Grab a small batch
    #####################
    data_iter = iter(data_loader)
    # Adjust if your dataset returns a different structure
    samples, _, mse_linear_weight = next(data_iter)

    # Limit the batch to `max_images`
    samples = samples[:max_images].to(device)
    mse_linear_weight = mse_linear_weight[:max_images].to(device)

    # We'll keep the original (denormalized) aside
    orig_denorm = denormalize(samples, mean, std)

    #####################
    # 2) For each checkpoint, load and run inference
    #####################
    recons_list = []
    # Keep a copy of your model's initial weights (if you plan to restore after)
    original_state = model.state_dict()

    for ckpt_path in checkpoint_paths:
        print(f"\n>>> Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print("Epoch in this checkpoint:", ckpt["epoch"])
        
        # If the checkpoint dict is structured as {'model': state_dict, ...}
        model.load_state_dict(ckpt['model'], strict=False)
        model.eval()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            if name.endswith("decoder_pred.weight"):
                print(f"Checkpoint={ckpt_path} => {name}[100,100]: {param[100,100].item():.12f}")
                break

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=True):
                # forward pass
                loss, x_rec, mask_s4 = model(samples, mse_linear_weight, mask_ratio=mask_ratio)

            wrapped_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            x_reconstructed = wrapped_model.unpatchify(x_rec)  # shape [B, C, H, W]
            recon_denorm = denormalize(x_reconstructed, mean, std)
            recons_list.append(recon_denorm)

    # (optional) restore original state if you intend to do something else with the model
    model.load_state_dict(original_state)

    #####################
    # 3) Build one combined image per sample
    #####################
    B = samples.shape[0]
    for i in range(B):
        # Start with the original image => shape [1, C, H, W]
        combo = orig_denorm[i : i+1]

        # For each reconstruction, add a white gap + that reconstruction
        for recon_denorm in recons_list:
            # White gap: all ones => shape [1, C, H, gap_width]
            gap_strip = torch.ones(
                (1, combo.shape[1], combo.shape[2], gap_width),
                device=combo.device,
                dtype=combo.dtype
            )
            # Add gap, then next reconstruction
            combo = torch.cat([combo, gap_strip, recon_denorm[i:i+1]], dim=3)

        # (Optional) upsample for nicer viewing:
        # e.g. if upsample_factor=2 => final dimension doubles
        if upsample_factor > 1:
            new_h = combo.shape[2] * upsample_factor
            new_w = combo.shape[3] * upsample_factor
            combo = F.resize(combo, [new_h, new_w])

        # Save
        out_path = os.path.join(output_dir, f"compare_sample_{i}.png")
        # 'normalize=True' will min/max scale each channel to [0..1]
        save_image(combo, out_path, nrow=1, normalize=True)
        print(f"Saved comparison to: {out_path}")

    print("All done!\n")


def run_inference_reconstruction(
    model, data_loader, device,
    output_dir, mean, std,
    max_images=30, mask_ratio=0.0
):
    """
    Run a forward pass on ~max_images samples to visualize reconstruction.

    model: the MixMIM model (already on `device`)
    data_loader: a DataLoader yielding (images, labels) or (images, _)
    device: CUDA device
    output_dir: where to write the 'reconstruction/' folder
    mean, std: channel-wise normalization stats
    max_images: how many samples to visualize
    mask_ratio: how many patches to mask during inference (0.0 => no masking)
    """
    model.eval()
    recon_dir = os.path.join(output_dir, "reconstruction")
    os.makedirs(recon_dir, exist_ok=True)

    images_saved = 0
    with torch.no_grad():
        for step, (samples, _, mse_linear_weight) in enumerate(data_loader):
            samples = samples.to(device, non_blocking=True)
            mse_linear_weight = mse_linear_weight.to(device, non_blocking=True)

            # Forward pass with AMP
            with torch.amp.autocast(device_type='cuda', enabled=True):
                #   loss, x_rec, mask_s4 = model(samples, samples, mask_ratio=mask_ratio)
                # model’s forward
                loss, x_rec, mask_s4 = model(samples, mse_linear_weight, mask_ratio=mask_ratio)

            # The model's "unpatchify" yields reconstructed images [B, C, H, W]
            x_reconstructed = model.unpatchify(x_rec)

            # Both `samples` and `x_reconstructed` are [B, 2, H, W] in your 2‐channel case
            orig_denorm = denormalize(samples, mean, std)
            rec_denorm  = denormalize(x_reconstructed, mean, std)

            B = samples.shape[0]
            num_to_save = min(B, max_images - images_saved)

            for i in range(num_to_save):
                # Single sample
                o_ = orig_denorm[i : i+1]  # shape [1, C=2, H, W]
                r_ = rec_denorm[i : i+1]   # shape [1, C=2, H, W]

                # Combine side-by-side along width dimension => [1, 2, H, 2W]
                combo = torch.cat([o_, r_], dim=3)

                out_file = os.path.join(recon_dir, f"sample_{images_saved + i}.png")
                # `save_image` with `normalize=True` will re-scale to [0..1] for visualization
                save_image(combo, out_file, nrow=1, normalize=True)

            images_saved += num_to_save
            if images_saved >= max_images:
                break

    print(f"Inference images saved to: {recon_dir}")


def get_args_parser():
    parser = argparse.ArgumentParser('MixMIM pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mixmim_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=128, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)  

    # NEW ARGUMENT: to allow skipping training for reconstruction/inference
    parser.add_argument('--inference', action='store_true',
                        help='If set, run reconstruction inference and skip training.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--port', default=29529, type=int)

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
    
    # Mean/std for the BigEarthNetv2 data
    benv2_mean = [-19.1966, -12.5144]
    benv2_std  = [5.4501,   5.0019]

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.Resize(size=(args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=benv2_mean, std=benv2_std)])
    
    dataset_train = BigEarthNetv2(csv_path='datasets/benv2_train_val.csv', root_dir=args.data_path, transform=transform_train)
    print(dataset_train)

    global_rank = misc.get_rank()

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
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
        persistent_workers=False,  # turn on when memory is sufficient
    )

    # define the model
    model = create_model(
        args.model,
        norm_pix_loss=args.norm_pix_loss,
    )

    if args.finetune:
        # add support for ceph
        if args.finetune.startswith('s3:'):
            from petrel_client.client import Client
            client = Client()
            with io.BytesIO(client.get(args.finetune)) as f:
                checkpoint = torch.load(f, map_location='cpu')
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    if global_rank == 0:
        model.eval()
        in_chans = 2
        flops = FlopCountAnalysis(model, (torch.rand(1, in_chans, args.input_size, args.input_size).to(device), torch.rand(1, in_chans, args.input_size, args.input_size).to(device)))
        print(flop_count_table(flops, max_depth=2))
        model.train()

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

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

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    # ------------------------------------------------------------------
    # If --inference is set, just run the reconstruction & exit
    # ------------------------------------------------------------------
    if args.inference:
        epochs_to_compare = [64, 128, 256, 600, 1024]
        ckpt_paths = build_checkpoint_paths_from_resume(args, epochs_to_compare)
        print("\n***** Inference-only mode: skipping training *****\n")
        compare_checkpoints_reconstruction(
            model,
            ckpt_paths,
            data_loader=data_loader_train,
            device=device,
            output_dir="reconstruction",
            mean=benv2_mean,
            std=benv2_std,
            max_images=50,       # show 5 samples
            mask_ratio=0.0,     # partial masking
            gap_width=5,        # 5-pixel white gap between columns
            upsample_factor=3   # enlarge final image by 2x
        )
        return  # exit the script here

    # ------------------------------------------------------------------
    # Otherwise, proceed with the usual training loop
    # ------------------------------------------------------------------

    # save to ceph
    use_ceph = args.resume.startswith('s3:')

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            model.module.cur_epoch = epoch    
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, use_ceph=use_ceph)
        
        if (epoch+1 == 64):
            checkpoint_file = "checkpoint_{}.pth".format(epoch+1)
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            checkpoint_path = os.path.join(args.output_dir, checkpoint_file)
            misc.save_on_master(to_save, checkpoint_path)
            print(f"Model saved at epoch {epoch} to the file {checkpoint_file}")
            
            break

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

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
