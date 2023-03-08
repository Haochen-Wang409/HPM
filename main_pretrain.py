# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# UM-MAE: https://github.com/implus/UM-MAE
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import builtins

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  
import timm.optim.optim_factory as optim_factory
from timm.utils import ModelEma

from util import utils
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import ImageListFolder

from engine_pretrain import train_one_epoch
from mask_transform import MaskTransform

import models_mae
import models_mae_learn_loss
import models_mae_learn_feature_loss


def get_args_parser():
    parser = argparse.ArgumentParser('Hard Patches Mining for Masked Image Modeling', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--bf16', action='store_true', help='whether to use bf16')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--token_size', default=int(224 / 16), type=int,
                        help='number of patch (in one dimension), usually input_size//16')  # for mask generator
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')

    # Mask parameters (by UM-MAE)
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_regular', action='store_true',
                        help='Uniform sampling for supporting pyramid-based vits')
    parser.set_defaults(mask_regular=False)
    parser.add_argument('--mask_block', action='store_true',
                        help='Block sampling for supporting pyramid-based vits')
    parser.set_defaults(mask_block=False)
    parser.add_argument('--vis_mask_ratio', default=0.0, type=float,
                        help='Secondary masking ratio (mask percentage of visible patches, secondary masking phase).')

    # HPM parameters
    parser.add_argument('--learning_loss', action='store_true', help='Learn to predict loss for each patch.')
    parser.set_defaults(learning_loss=True)
    parser.add_argument('--learn_feature_loss', default='none', type=str,
                        help='Use MSE loss for features as target.')
    parser.add_argument('--relative', action='store_true', help='Use relative learning loss or not.')
    parser.set_defaults(relative=True)
    parser.add_argument('--dino_path', default='none', type=str,
                        help='Pre-trained DINO for feature distillation (ViT-B/16).')
    parser.add_argument('--clip_path', default='none', type=str,
                        help='Pre-trained CLIP for feature distillation (ViT-B/16).')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, 40 for MAE and 10 for SimMIM')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--load_from', default='', help='load pretrained checkpoint model')
    parser.add_argument('--experiment', default='exp', type=str, help='experiment name (for log)')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

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

    # simple augmentation
    transform_train = MaskTransform(args)

    # build dataset
    dataset_train = ImageListFolder(os.path.join(args.data_path, 'train'), transform=transform_train,
                                    ann_file=os.path.join(args.data_path, 'train.txt'))
    print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if global_rank == 0 and args.log_dir is not None:
        log_dir = os.path.join(args.log_dir, f"{args.model}_{args.experiment}")
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    model_teacher = None
    # define the model
    if args.learning_loss:
        if args.learn_feature_loss != 'none':
            assert args.learn_feature_loss in ['clip', 'dino', 'ema']

            model = models_mae_learn_feature_loss.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                                       vis_mask_ratio=args.vis_mask_ratio)
            
            if args.learn_feature_loss == 'dino':
                model_teacher = timm.models.vit_base_patch16_224()
                model_teacher.load_state_dict(torch.load(args.dino_path), strict=False)
            else:
                from models_clip import build_model
                state_dict = torch.load(args.clip_path, map_location='cpu')
                model_clip = build_model(state_dict)
                model_clip.load_state_dict(state_dict, strict=False)
                model_clip.float()

                model_teacher = model_clip.visual

            model_teacher.to(device)
            model_teacher.eval()
        else:
            model = models_mae_learn_loss.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                               vis_mask_ratio=args.vis_mask_ratio)

    else:
        if args.learn_feature_loss != 'none':
            assert args.learn_feature_loss in ['clip', 'dino', 'ema']

            model = models_mae_learn_feature_loss.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                                       vis_mask_ratio=args.vis_mask_ratio,
                                                                       learning_loss=False)

            if args.learn_feature_loss == 'dino':
                model_teacher = timm.models.vit_base_patch16_224()
                model_teacher.load_state_dict(torch.load(args.dino_path), strict=False)
            else:
                from models_clip import build_model
                state_dict = torch.load(args.clip_path, map_location='cpu')
                model_clip = build_model(state_dict)
                model_clip.load_state_dict(state_dict, strict=False)
                model_clip.float()

                model_teacher = model_clip.visual

            model_teacher.to(device)
            model_teacher.eval()
        else:
            model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                    vis_mask_ratio=args.vis_mask_ratio)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # define ema model
    model_ema = None
    if args.byol or args.learning_loss or args.learn_feature_loss == 'ema':
        # use momentum encoder for BYOL
        model_ema = ModelEma(model, decay=0.999, device=args.device, resume='')

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

    ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp.pth")
    if not os.path.isfile(ckpt_path):
        print("Checkpoint not founded in {}, train from random initialization".format(ckpt_path))
    else:
        print("Found checkpoint at {}".format(ckpt_path))
        misc.load_model(args=args, ckpt_path=ckpt_path, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, model_ema=model_ema.ema)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            model_ema=model_ema,
            model_teacher=model_teacher,
        )

        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            # "ema_state_dict": model_ema.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model,
        }
        if model_ema is not None:
            save_dict['ema_state_dict'] = model_ema.ema.state_dict()
        if loss_scaler is not None:
            save_dict['loss_scaler'] = loss_scaler.state_dict()

        ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp.pth")
        utils.save_on_master(save_dict, ckpt_path)
        print(f"model_path: {ckpt_path}")

        if args.output_dir and ((epoch + 1) % 100 == 0 or epoch + 1 == args.epochs):
            ckpt_path = os.path.join(args.output_dir,
                                     "{}_{}_{:04d}.pth".format(args.model, args.experiment,
                                                                     epoch))
            utils.save_on_master(save_dict, ckpt_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(
                    args.output_dir,
                    "{}_{}_log.txt".format(
                        args.model,
                        args.experiment
                    )
            ), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
