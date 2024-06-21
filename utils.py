# import apex.amp as amp
import math
import logging
import pdb
import time
import os
import sys
import torch
import argparse
import warnings
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import logging
# import appdirs
import pickle
from collections import defaultdict, namedtuple
from collections.abc import Sequence
import operator
import math
import warnings

from config import Train_Config
from torch.hub import load_state_dict_from_url
from typing import Tuple

logging.basicConfig(
    format='%(levelname)-8s %(asctime)-12s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('AUTOLIRPA_DEBUG', 0) else logging.INFO)

warnings.simplefilter("once")


def get_loaders(args):
    if 'CIFAR' in args.dataset:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = eval(f'datasets.{args.dataset}')(
            os.path.join(args.data_dir, args.dataset), train=True,
            transform=train_transform, download=True
        )
        test_dataset = eval(f'datasets.{args.dataset}')(
            os.path.join(args.data_dir, args.dataset), train=False,
            transform=test_transform, download=True
        )
    elif 'ImageNet' in args.dataset:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.ImageFolder(
            root=f'{args.data_dir}/{args.dataset}/train', transform=train_transform)
        test_dataset = datasets.ImageFolder(
            root=f'{args.data_dir}/{args.dataset}/val', transform=test_transform)
    else:
        raise ValueError(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size // args.accum_steps, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.num_workers,
    )
    print(f"pin_memory: {args.pin_memory}, num_workers: {args.num_workers}")
    return train_loader, test_loader


def get_lr_schedular(total_steps, lr_base, decay_type, warmup_steps, linear_end=1e-5):
    lrs = []
    for step in range(total_steps):
        if step <= warmup_steps:
            lr = lr_base * np.minimum(1., step / warmup_steps)
            lrs.append(lr)
        else:
            progress = (step - warmup_steps) / float(total_steps - warmup_steps)
            progress = np.clip(progress, 0.0, 1.0)
            if decay_type == 'linear':
                lr = linear_end + (lr_base - linear_end) * (1.0 - progress)
            elif decay_type == 'cosine':
                lr = lr_base * 0.5 * (1. + np.cos(np.pi * progress))
            else:
                raise ValueError(f'Unknown lr type {decay_type}')
            lrs.append(lr)
    return lrs


class MultiAverageMeter(object):
    """Computes and stores the average and current value for multiple metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_meter = defaultdict(float)
        self.lasts = defaultdict(float)
        self.counts_meter = defaultdict(int)

    def update(self, key, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.lasts[key] = val
        self.sum_meter[key] += val * n
        self.counts_meter[key] += n

    def last(self, key):
        return self.lasts[key]

    def avg(self, key):
        if self.counts_meter[key] == 0:
            return 0.0
        else:
            return self.sum_meter[key] / self.counts_meter[key]

    def __repr__(self):
        s = ""
        for k in self.sum_meter:
            s += "{}={:.4f} ".format(k, self.avg(k))
        return s.strip()


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def load_pretrained_model_from_path(model, _logger, nci=True, cfg=None, num_classes=1000,
                                    in_chans=3, filter_fn=None, strict=True, progress=False):
    cfg = cfg or getattr(model, 'default_cfg')
    if cfg is None or not cfg.get('url', None):
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    # _logger.warning(f'nci={nci}')
    # if nci:

    args_vit = Train_Config(get_args())
    path = f'./checkpoints/pretrained/{args_vit.model}.pth'
    # path_21k = './checkpoints/pretrained/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
    state_dict = torch.load(path, map_location='cpu')
    _logger.warning('Use local pretrained checkpoint!')
    # else:
    #     _logger.warning('Do not use local pretrained checkpoint!!!')
    #     state_dict = load_state_dict_from_url(cfg['url'], progress=progress, map_location='cpu')

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    input_convs = cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifier_name = cfg['classifier']
    label_offset = cfg.get('label_offset', 0)
    if num_classes != cfg['num_classes']:
        # completely discard fully connected if model num_classes doesn't match pretrained weights
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False
    elif label_offset > 0:
        # special case for pretrained weights with an extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=False)


def get_named_paras(net):
    for name, parameters in net.named_parameters():
        print(name, ': ', parameters.size())


def get_args():
    parser = argparse.ArgumentParser()
    # random seed
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    # model
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--scratch', action="store_true")
    parser.add_argument('--proj_dir', type=str, default='/home/hhgong/code/vit_rar')
    # dataset
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--crop', type=int, default=32)
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=4, help='swin transformer')
    # training
    parser.add_argument('--train_mode', type=str, default='nat', choices=['nat', 'fgsm', 'pgd', 'trades'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--attack_iters', default=10, type=int)
    parser.add_argument('--lr_max', default=0.1, type=float)
    # gpu
    parser.add_argument('--mul_gpus', action="store_true")
    # random setting
    parser.add_argument('--r_std', default=0.0, type=float)
    parser.add_argument('--r_steps', default=10, type=int)
    parser.add_argument('--r_alpha', default=0.01, type=float)
    # eval
    parser.add_argument('--load', type=str)
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--test_mode', type=str, default='nat')
    parser.add_argument('--n_queries', type=int, default=100)

    args = parser.parse_args()

    return args


def get_opt(model, args):
    return torch.optim.SGD(
        model.parameters(), lr=args.lr_max,
        momentum=args.momentum, weight_decay=args.weight_decay
    )


def lr_schedule(t, args):
    if args.dataset == "ImageNet":
        if args.epochs == 20:
            # args.lr_max = 0.01
            if t < 10:
                return args.lr_max
            elif t < 15:
                return args.lr_max * 0.1
            else:
                return args.lr_max * 0.01
        elif args.epochs == 10:
            if t < 5:
                return args.lr_max
            elif t < 8:
                return args.lr_max * 0.1
            else:
                return args.lr_max * 0.01
    else:  # args.dataset in ['CIFAR10', 'CIFAR100', 'ImageNette']:
        # args.lr_max = 0.1
        if args.epochs == 40:
            if t < 20:
                return args.lr_max
            elif t < 30:
                return args.lr_max * 0.1
            else:
                return args.lr_max * 0.01
        elif args.epochs == 50:
            if t < 25:
                return args.lr_max
            elif t < 40:
                return args.lr_max * 0.1
            else:
                return args.lr_max * 0.01
        else:
            return args.lr_max
    raise ValueError(args.dataset)
