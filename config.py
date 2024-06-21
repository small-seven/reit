import os
import torch
import argparse
import random
import numpy as np
import torch.backends.cudnn as cudnn


def simple_fix_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def fix_random_seed(seed):
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置python哈希种子，为了禁止hash随机化
    torch.manual_seed(seed)  # torch cpu
    torch.cuda.manual_seed(seed)  # torch gpu
    torch.cuda.manual_seed_all(seed)  # torch all gpus
    cudnn.benchmark = False  #
    cudnn.deterministic = True


cfg_datasets = {
    'CIFAR10': dict(
        data_mean=(0.4914, 0.4822, 0.4465),
        data_std=(0.2471, 0.2435, 0.2616),
        num_classes=10,
    ),
    'SVHN': dict(
        data_mean=(0.4377, 0.4438, 0.4728),
        data_std=(0.1980, 0.2010, 0.1970),
        num_classes=10,
    ),
    'TinyImageNet': dict(
        data_mean=(0.485, 0.456, 0.406),
        data_std=(0.229, 0.224, 0.225),
        num_classes=200,
    ),
    'CIFAR100': dict(
        data_mean=(0.5071, 0.4867, 0.4408),
        data_std=(0.2675, 0.2565, 0.2761),
        num_classes=100,
    ),
    'ImageNet': dict(
        data_mean=(0.485, 0.456, 0.406),
        data_std=(0.229, 0.224, 0.225),
        num_classes=1000,
    ),
    'ImageNette': dict(
        data_mean=(0.485, 0.456, 0.406),
        data_std=(0.229, 0.224, 0.225),
        num_classes=10,
    ),

}


class Train_Config(object):
    # random seed
    seed = 123

    # model setting
    model = 'vit_base_patch16_224'  # vit_base_patch32_384
    depth = 12
    num_layers = 12
    load = ''
    scratch = False
    proj_dir = '/home/hhgong/code/vit_rar'

    # data setting
    dataset = 'CIFAR10'
    data_dir = '../data'
    patch = 4
    crop = 32
    resize = 32

    # training setting
    train_mode = 'nat'  # choices=['nat', 'fgsm', 'pgd', 'trades']
    batch_size = 128
    accum_steps = 1
    grad_clip = 1.0
    start_epoch = 0
    epochs = 40
    lr_max = 0.1
    lr_min = 0.
    weight_decay = 1e-4
    momentum = 0.9

    # adversarial setting
    attack_iters = 10
    epsilon = 8
    alpha = 2  # step size

    # random setting
    r_std = 0.1
    r_steps = 10
    r_alpha = 0.01

    # eval setting
    eval = False
    test_mode = 'nat'
    c = 1e-4  # cwl2
    steps = 1000  # cwl2

    # log setting
    log_dir = './logs'
    log_interval = 100

    # saving setting
    saved_dir = './checkpoints'

    # gpu setting
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mul_gpus = False
    pin_memory = False
    num_workers = 8

    # other setting
    run_dummy = False
    prefetch = 1  # 2
    tfds_dir = '~/dataset/tar'
    no_timm = False
    data_loader = 'torch'
    no_inception_crop = False
    custom_vit = False
    load_state_dict_only = False
    patch_embed_scratch = False
    downsample_factor = False
    pretrain_pos_only = False

    def __init__(self, args=None):
        if args is not None:
            names = self.__dict__
            for arg in vars(args):
                if arg == 'device':
                    names[arg] = f'cuda:{getattr(args, arg)}'
                else:
                    names[arg] = getattr(args, arg)

        # dataset setting
        self.data_mean = cfg_datasets[self.dataset]['data_mean']
        self.data_std = cfg_datasets[self.dataset]['data_std']
        self.num_classes = cfg_datasets[self.dataset]['num_classes']

        simple_fix_random_seed(args.seed)
