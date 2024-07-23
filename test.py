import argparse
import logging
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Train_Config
from utils import get_loaders, get_args, MultiAverageMeter, logger
import torchattack
import warnings

warnings.filterwarnings("ignore")


def evaluate_attack(args, model, test_loader, atk, atk_name):
    model.eval()
    # load = "${net}_${dataset}_${train_mode}_r${r_std}.pth"
    if atk_name == "autoattack":
        test_ckpt_path = f"./test_ckpt/{args.load.replace('.pth', '')}" \
                         f"_{atk_name}_b{args.batch_size}.pth"
    else:
        test_ckpt_path = None
    try:
        test_ckpt = torch.load(test_ckpt_path)
        meter = test_ckpt['meter']
        batch_idx = test_ckpt['batch_idx']
    except:
        meter = MultiAverageMeter()
        batch_idx = -1
    # print(meter)
    # print(batch_idx)
    # exit()
    start_time = time.time()
    for i, (X, y) in enumerate(test_loader):
        if i <= batch_idx: continue
        if i == batch_idx + 1 and i != 0:
            logger.info(f"Testing begins with batch_idx={i}... {test_ckpt_path}")
        if atk_name == "autoattack":
            if i % 50 == 0: logger.info(i)
        X, y = X.to('cuda'), y.to('cuda')

        X_adv = atk(X, y)  # advtorch
        if args.r_std != 0.0:
            output = model(X_adv)
        else:
            with torch.no_grad():
                output = model(X_adv)
        with torch.no_grad():
            loss = F.cross_entropy(output, y)
            meter.update('test_loss', loss.item(), y.size(0))
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))
            if atk_name == "autoattack":
                os.makedirs("./test_ckpt/", exist_ok=True)
                test_ckpt = {"meter": meter, "batch_idx": i}
                torch.save(test_ckpt, test_ckpt_path)

    elapsed = int(time.time() - start_time)
    hours = elapsed // 3600
    minutes = (elapsed - 3600 * hours) // 60
    logger.info(f'Attack_type: [{atk_name}] done, acc: {float(meter.avg("test_acc")): .4f}, '
                f'Elapsed Time: {hours}h, {minutes}min')


def main(args):
    # set current device
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    print(args.__dict__)
    _, test_loader = get_loaders(args)

    if args.model in ['vit_small_patch16_224', 'vit_base_patch16_224',
                      'vit_tiny_patch16_224']:
        from timm_vit.vit_rar import vit_small_patch16_224, vit_base_patch16_224, \
            vit_tiny_patch16_224
        model = eval(args.model)(
            pretrained=(not args.scratch),
            img_size=args.crop, num_classes=args.num_classes,
            patch_size=args.patch, args=args
        ).cuda()
    elif args.model in ['deit_tiny_patch16_224', 'deit_small_patch16_224']:
        from timm_vit.deit_rar import deit_tiny_patch16_224, deit_small_patch16_224
        model = eval(args.model)(
            pretrained=(not args.scratch),
            img_size=args.crop, num_classes=args.num_classes,
            patch_size=args.patch, args=args
        ).cuda()
    elif args.model in ['swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
                        'swin_base_patch4_window7_224']:
        from timm_vit.swin_rar import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, \
            swin_base_patch4_window7_224
        model = eval(args.model)(
            pretrained=(not args.scratch),
            img_size=args.crop, num_classes=args.num_classes,
            patch_size=args.patch, window_size=args.window_size, args=args
        ).cuda()
    else:
        raise ValueError(args.model)

    if not args.eval:
        raise ValueError(args.eval)

    if args.load:
        print(args.load)
        load_folder = f'./checkpoints/{args.dataset}'
        checkpoint = torch.load(f'{load_folder}/{args.load}')
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        raise ValueError(args.load)

    model.eval()
    if args.mul_gpus:
        model = torch.nn.DataParallel(model)

    # attack strength setting
    if args.dataset == "ImageNet":
        eps, alpha = 4 / 255., 1 / 255.
    else:
        eps, alpha = 8 / 255., 2 / 255.

    if args.test_mode not in ['nat', 'fgsm', 'mifgsm', 'pgd', 'deepfool', 'cwl2', 'autoattack', 'all']:
        raise ValueError(args.test_mode)
    if args.test_mode in ['nat', 'all']:
        atk = torchattack.VANILA(model)
        evaluate_attack(args, model, test_loader, atk, 'vanilla')
    if args.test_mode in ['fgsm', 'all']:
        atk = torchattack.FGSM(model, eps=eps)
        evaluate_attack(args, model, test_loader, atk, 'fgsm')
    if args.test_mode in ['mifgsm', 'all']:
        atk = torchattack.MIFGSM(model, eps=eps, alpha=alpha, steps=5, decay=1.0)
        evaluate_attack(args, model, test_loader, atk, 'mifgsm')
    if args.test_mode in ['pgd', 'all']:
        atk = torchattack.PGD(model, eps=eps, alpha=alpha, steps=10, random_start=True)
        evaluate_attack(args, model, test_loader, atk, 'pgd')
    if args.test_mode in ['deepfool']:
        atk = torchattack.DeepFool(model, steps=50, overshoot=0.02)
        evaluate_attack(args, model, test_loader, atk, 'deepfool')
    if args.test_mode in ['cwl2', 'all']:
        atk = torchattack.CW(model, c=args.c, kappa=0, steps=args.steps, lr=0.01)
        evaluate_attack(args, model, test_loader, atk, 'cwl2')
    if args.test_mode in ['autoattack']:
        atk = torchattack.AutoAttack(model, norm='Linf', eps=eps, version='standard', n_classes=args.num_classes,
                                     n_queries=args.n_queries)
        evaluate_attack(args, model, test_loader, atk, 'autoattack')


if __name__ == "__main__":
    main(Train_Config(get_args()))
