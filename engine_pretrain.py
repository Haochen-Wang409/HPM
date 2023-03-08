# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# UM-MAE: https://github.com/implus/UM-MAE
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import builtins

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, args=None, model_ema=None, model_teacher=None):
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 20

    accum_iter = args.accum_iter

    if args.learning_loss:
        assert model_ema is not None
        if epoch < 100:
            model_ema.decay = 0.999 + epoch / 100 * (0.9999 - 0.999)
        else:
            model_ema.decay = 0.9999

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = len(data_loader) * epoch + data_iter_step
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images, bool_masked_pos = batch

        samples = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)   # (N, L)
        visible_mask = torch.zeros_like(bool_masked_pos).to(device, non_blocking=True).to(torch.bool)

        with torch.cuda.amp.autocast():
            if model_ema is not None:
                with torch.no_grad():
                    outs_ema = model_ema.ema(samples, mask=visible_mask)

            if args.learning_loss:
                # generate mask by predicted loss
                mask = model_ema.ema.generate_mask(outs_ema['loss_pred'], mask_ratio=args.mask_ratio,
                                                   guide=True, epoch=epoch, total_epoch=args.epochs)
                bool_masked_pos = mask.to(device, non_blocking=True).flatten(1).to(torch.bool)

            outs = model(samples, mask=bool_masked_pos)

            if args.learn_feature_loss != 'none':
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        if args.learn_feature_loss in ['clip', 'dino']:
                            feature_target = forward_features(model_teacher, samples, args.learn_feature_loss)
                            
                        elif args.learn_feature_loss == 'ema':
                            feature_target = outs_ema['features'][:, 1:, :]

                loss_outs = model.module.forward_loss(
                    outs['pix_pred'][:, -outs['mask_num']:],
                    feature_target.detach(),
                    outs['mask'],
                )
            else:
                loss_outs = model.module.forward_loss(
                    samples,
                    outs['pix_pred'][:, -outs['mask_num']:],
                    outs['mask'],
                )

            if isinstance(loss_outs, dict):
                loss = loss_outs['mean']
            else:
                loss = loss_outs

            if args.learning_loss:
                loss_target = loss_outs['matrix']

                loss_learn = model.module.forward_learning_loss(
                    outs['loss_pred'][:,  -outs['mask_num']:],
                    bool_masked_pos,
                    loss_target.detach(),
                    relative=args.relative,
                )
                loss_learn_value = loss_learn.item()
                if not math.isfinite(loss_learn_value):
                    print("Loss learning is {}, skip".format(loss_learn_value))
                    sys.exit(1)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, skip".format(loss_value))
            sys.exit(1)

        if args.learning_loss:
            loss += loss_learn

        loss /= accum_iter
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if args.learning_loss:
            metric_logger.update(loss_learn=loss_learn_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(grad_norm=grad_norm)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if args.learning_loss:
            loss_learn_value_reduce = misc.all_reduce_mean(loss_learn_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            log_writer.add_scalar('train_loss', loss_value_reduce, it)
            log_writer.add_scalar('lr', lr, it)
            log_writer.add_scalar('grad_norm', grad_norm, it)

            if args.learning_loss:
                log_writer.add_scalar('train_loss_learn', loss_learn_value_reduce, it)

        if (data_iter_step + 1) >= len(data_loader):
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def forward_features(model, x, model_type):
    assert model_type in ['dino', 'clip']
    if model_type == 'dino':
        return forward_features_dino(model, x)
    else:
        return forward_features_clip(model, x)


def forward_features_dino(model, x):
    B = x.shape[0]
    x = model.patch_embed(x)

    cls_tokens = model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    for blk in model.blocks:
        x = blk(x)

    x = model.norm(x)
    return x[:, 1:, :]


def forward_features_clip(model, x):
    x = model.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
        dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.positional_embedding.to(x.dtype)
    x = model.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # x = model.ln_post(x[:, 0, :])
    x = model.ln_post(x)

    if model.proj is not None:
        x = x @ model.proj

    return x[:, 1:, :]