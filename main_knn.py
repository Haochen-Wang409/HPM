# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DINO: https://github.com/facebookresearch/dino
# --------------------------------------------------------

import argparse
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import timm

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.datasets import build_dataset

import models_vit


def get_args_parser():
    parser = argparse.ArgumentParser('Hard Patches Mining for Masked Image Modeling', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
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
    parser.add_argument('--num_workers', default=4, type=int)
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

    # store features
    parser.add_argument('--dump_features', default=None,
                        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None,
                        help='If the features have already been computed, where to find them.')

    return parser


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    metric_logger = misc.MetricLogger(delimiter="  ")
    features = None
    for idx, (samples, index) in enumerate(metric_logger.log_every(data_loader, 20)):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model.module.forward_features(samples).clone()
        # print('feats:', feats.shape, torch.unique(feats[0])[:5])
        # print('feats_norm:', torch.unique(torch.nn.functional.normalize(feats[0], dim=0, p=2))[:5])

        # init storage feature matrix
        if misc.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # # get indexes from all processes
        # y_all = torch.empty(misc.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        # y_l = list(y_all.unbind(0))
        # y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        # y_all_reduce.wait()
        # index_all = torch.cat(y_l)
        #
        # # share features between processes
        # feats_all = torch.empty(
        #     misc.get_world_size(),
        #     feats.size(0),
        #     feats.size(1),
        #     dtype=feats.dtype,
        #     device=feats.device,
        # )
        # output_l = list(feats_all.unbind(0))
        # output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        # output_all_reduce.wait()
        #
        # # update storage feature matrix
        # if misc.get_rank() == 0:
        #     if use_cuda:
        #         features.index_copy_(0, index_all, torch.cat(output_l))
        #     else:
        #         features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())

        features[idx * len(index) : (idx + 1) * len(index), :] = feats

    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 500
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        if len(torch.unique(test_features[idx : min((idx + imgs_per_chunk), num_test_images), :])) == 0:
            print('warning, test features are all 0!')
        if len(torch.unique(train_features[:, idx : min((idx + imgs_per_chunk), num_test_images)])) == 0:
            print('warning, train_features are all 0!')

        similarity = torch.mm(features, train_features)                         # [B, N]
        distances, indices = similarity.topk(k, largest=True, sorted=True)      # [B, K]
        candidates = train_labels.view(1, -1).expand(batch_size, -1)            # [B, N]
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))  # [B, C]
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


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

    if args.load_features:
        print(f"Load features from {args.load_features}")
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)

        if args.distributed:  # args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print(
                        'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank,
                    shuffle=True)  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

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

        model = models_vit.__dict__[args.model](
            num_classes=0,
            global_pool=False,
        )

        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            if 'state_dict' in checkpoint.keys():
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            checkpoint_model = {}

            for name, param in state_dict.items():
                if name.startswith('module.'):
                    checkpoint_model[name[7:]] = param
                else:
                    checkpoint_model[name] = param

            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)

            # if args.global_pool:
            #     assert set(missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer: following MoCo v3
            # trunc_normal_(model.head.weight, std=0.01)

        model.to(device)
        model.eval()

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        # ============ extract labels ... ============
        train_labels = torch.tensor([s for s in dataset_train.target]).long()
        test_labels = torch.tensor([s for s in dataset_val.target]).long()

        # ============ extract features ... ============
        print("Extracting features for train set...")
        train_features = extract_features(model, data_loader_train)
        print("Extracting features for val set...")
        test_features = extract_features(model, data_loader_val)

        if misc.get_rank() == 0:
            num_train = len(train_features)
            train_features = torch.cat((
                torch.nn.functional.normalize(train_features[: num_train // 2], dim=1, p=2),
                torch.nn.functional.normalize(train_features[num_train // 2 :], dim=1, p=2),
            ), dim=0)
            test_features = torch.nn.functional.normalize(test_features, dim=1, p=2)

    # save features and labels
    if args.dump_features and misc.get_rank() == 0:
        os.makedirs(args.dump_features, exist_ok=True)
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))

    if misc.get_rank() == 0:
        train_features = train_features.detach().cuda()
        test_features = test_features.detach().cuda()
        train_labels = train_labels.detach().cuda()
        test_labels = test_labels.detach().cuda()

        print("Features are ready!\nStart the k-NN classification.")
        print("Pre-trained checkpoint from: %s" % args.finetune)
        for k in [10, 20, 100, 200]:
            top1, top5 = knn_classifier(train_features, train_labels,
                                        test_features, test_labels, k, 0.07)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)