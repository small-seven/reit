import logging
import math
from datetime import datetime
import torch.nn.functional as F
import numpy as np
import os

from thop import profile

import torch

from config import Train_Config
from utils import MultiAverageMeter, logger, get_lr_schedular, get_loaders, get_args, get_opt, lr_schedule
from torch.autograd import Variable
import torch.nn as nn

import warnings

import torchattack

warnings.filterwarnings("ignore")


def evaluate_pgd(args, model, test_loader):
    # attack strength setting
    if args.dataset == "ImageNet":
        eps, alpha = 4 / 255., 1 / 255.
    else:
        eps, alpha = 8 / 255., 2 / 255.

    model.eval()
    atk = torchattack.PGD(model, eps=eps, alpha=alpha, steps=10, random_start=True)
    meter = MultiAverageMeter()
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        X_adv = atk(X, y)
        if args.eval:
            output = model(X_adv)
            loss = F.cross_entropy(output, y)
        else:
            with torch.no_grad():
                output = model(X_adv)
                loss = F.cross_entropy(output, y)
        meter.update('test_loss', loss.item(), y.size(0))
        meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

    return meter.avg('test_loss'), meter.avg('test_acc')


def train_adv(args, model, train_loader, test_loader):
    # x = torch.randn(1, 3, 32, 32)
    x = torch.randn(1, 3, 224, 224)
    # device1 = torch.device("cuda:0")
    # model.to(device1)
    flops, params = profile(model, inputs=(x,))
    print(args.model, x.size())
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    exit(0)

    # attack strength setting
    if args.dataset == "ImageNet":
        eps, alpha = 4 / 255., 1 / 255.
    else:
        eps, alpha = 8 / 255., 2 / 255.

    criterion = nn.CrossEntropyLoss()

    opt = get_opt(model, args)

    def evaluate():
        pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader)
        logger.info(f'PGD testing : loss {pgd_loss:.4f} acc {pgd_acc:.4f}')
        opt.zero_grad()
        return pgd_acc

    best_acc = 0.
    evaluate_natural(args, model, test_loader)
    model.train()
    for epoch in range(args.start_epoch + 1, args.epochs + 1):

        train_loss, train_acc, train_n = 0, 0, 0
        lr = lr_schedule(epoch, args)
        logger.info(f'Epoch {epoch}, lr={lr:.6f}')
        opt.param_groups[0].update(lr=lr)

        def train_step(X, y):
            if args.train_mode == 'pgd':
                model.eval()
                atk = torchattack.PGD(model, eps=eps, alpha=alpha, steps=args.attack_iters, random_start=True)
                X_adv = atk(X, y)
                model.train()
                output = model(X_adv)
                loss = criterion(output, y)
            elif args.train_mode == 'trades':
                beta = 6.0
                batch_size = len(X)
                delta = torch.zeros_like(X).cuda()
                delta.data = torch.clamp(delta, min=-eps, max=eps)
                x_adv = X.detach() + delta.detach()

                criterion_kl = nn.KLDivLoss(size_average=False)
                model.eval()

                for _ in range(args.attack_iters):
                    x_adv.requires_grad_()
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(X), dim=1))
                    grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                    with torch.no_grad():
                        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())

                        # projected into linf ball
                        eps_tensor = eps * torch.ones_like(X).cuda()

                        x_adv = torch.min(x_adv, X + eps_tensor)
                        x_adv = torch.max(x_adv, X - eps_tensor)

                        # image reasonability
                        x_adv = torch.clamp(x_adv, min=0., max=1.)

                model.train()

                x_adv = Variable(x_adv, requires_grad=False)

                output = logits = model(X)
                loss_natural = F.cross_entropy(logits, y)
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                                F.softmax(model(X), dim=1))
                loss = loss_natural + beta * loss_robust
            elif args.train_mode == 'fgsm':
                model.eval()
                atk = torchattack.FGSM(model, eps=eps)
                X_adv = atk(X, y)
                model.train()
                output = model(X_adv)
                loss = criterion(output, y)
            else:
                raise ValueError(args.train_mode)

            (loss / args.accum_steps).backward()
            acc = (output.max(1)[1] == y).float().mean()

            return loss, acc

        for step, (X, y) in enumerate(train_loader):
            batch_size = args.batch_size // args.accum_steps
            for t in range(args.accum_steps):
                X_ = X[t * batch_size:(t + 1) * batch_size].cuda()  # .permute(0, 3, 1, 2)
                y_ = y[t * batch_size:(t + 1) * batch_size].cuda()  # .max(dim=-1).indices
                if len(X_) == 0:
                    break
                loss, acc = train_step(X_, y_)
                train_loss += loss.item() * y_.size(0)
                train_acc += acc.item() * y_.size(0)
                train_n += y_.size(0)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()
            opt.zero_grad()

            if (step + 1) % 500 == 0:
                logger.info(f"Training epoch {epoch} step {step + 1}/{len(train_loader)}, "
                            f"lr {opt.param_groups[0]['lr']:.4f} loss {train_loss / train_n:.4f} "
                            f"acc {train_acc / train_n:.4f}")

        test_acc = evaluate()
        if test_acc > best_acc:
            best_acc = test_acc
            saved_path = save_ckpt(args, test_acc, model, epoch, opt)
            logger.info(f'| Saving best model to {saved_path} ...\n')


def evaluate_natural(args, model, test_loader):
    model.eval()

    # with torch.no_grad():
    meter = MultiAverageMeter()

    def test_step(X_batch, y_batch):
        X, y = X_batch.cuda(), y_batch.cuda()
        if args.train_mode in ['nat', 'pgd', "trades", 'fgsm']:
            output = model(X)
            loss = F.cross_entropy(output, y)
        else:
            raise NotImplementedError(args.train_mode)
        meter.update('test_loss', loss.item(), y.size(0))
        meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

    if not args.eval:
        with torch.no_grad():
            for step, (X_batch, y_batch) in enumerate(test_loader):
                test_step(X_batch, y_batch)
    else:
        for step, (X_batch, y_batch) in enumerate(test_loader):
            test_step(X_batch, y_batch)

    logger.info('Evaluation nat {}'.format(meter))
    return meter.avg('test_acc')


def train_natural(args, model, train_loader, test_loader):
    opt = get_opt(model, args)

    meter = MultiAverageMeter()

    # if args.load:
    #     checkpoint = torch.load(f'{args.saved_dir}/{args.load}')
    #     args.start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['model'])
    #     if 'opt' in checkpoint:
    #         opt.load_state_dict(checkpoint['opt'])
    #     else:
    #         logging.warning('No state_dict for optimizer in the checkpoint')
    #     logging.info('Checkpoint resumed at epoch {}'.format(args.start_epoch))

    def train_step(epoch, step, X_batch, y_batch):
        # opt.param_groups[0]['lr'] = lr
        model.train()

        batch_size = math.ceil(args.batch_size / args.accum_steps)
        for i in range(args.accum_steps):
            X = X_batch[i * batch_size:(i + 1) * batch_size].cuda()
            y = y_batch[i * batch_size:(i + 1) * batch_size].cuda()

            if args.train_mode == 'nat':
                output = model(X)
                loss = F.cross_entropy(output, y)
                (loss * (X.shape[0] / X_batch.shape[0])).backward()
            else:
                raise NotImplementedError(args.train_mode)

            meter.update('train_loss', loss.item(), y.size(0))
            meter.update('train_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        opt.step()
        opt.zero_grad()

        # if step % args.log_interval == 0:
        #     logger.info(f'Training epoch {epoch} step {step} lr {lr:.4f} {meter}')

    # total_steps = args.epochs * len(train_loader)
    # lr_schedular = get_lr_schedular(total_steps, args.base_lr, 'cosine', args.warmup_steps)

    step, best_acc = 0, 0.
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        lr = lr_schedule(epoch, args)
        logger.info(f'Epoch {epoch}, lr={lr:.6f}')
        opt.param_groups[0].update(lr=lr)
        for (X_batch, y_batch) in train_loader:
            step += 1
            train_step(epoch, step, X_batch, y_batch)
        test_acc = evaluate_natural(args, model, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc
            saved_path = save_ckpt(args, test_acc, model, epoch, opt)
            logger.info(f'| Saving best model to {saved_path} ...\n')


def save_ckpt(args, test_acc, model, epoch, opt):
    if not os.path.exists(args.saved_dir):
        os.mkdir(args.saved_dir)
    if "vit" in args.model:
        model_name = "vit"
    elif "deit" in args.model:
        model_name = "deit"
    elif "swin" in args.model:
        model_name = "swin"
    else:
        raise ValueError(args.model)
    if 'small' in args.model:
        model_name = f'{model_name}_small'
    elif 'tiny' in args.model:
        model_name = f'{model_name}_tiny'
    elif 'base' in args.model:
        model_name = f'{model_name}_base'
    else:
        raise ValueError(args.model)

    saved_path = os.path.join(args.saved_dir,
                              f'{model_name}_{args.dataset}_{args.train_mode}_r{args.r_std}_lr{args.lr_max}.pth')
    state = {'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
             'epoch': epoch, 'opt': opt, 'test_acc': test_acc}
    torch.save(state, saved_path)

    return saved_path


def main(args):
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    logfile = os.path.join(args.log_dir,
                           f'{datetime.now().strftime("%Y-%m-%d")}vit_{args.dataset}_{args.train_mode}.log')

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
    logger.addHandler(file_handler)

    logger.info(args.__dict__)

    train_loader, test_loader = get_loaders(args)

    if args.model in ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224']:
        from timm_vit.vit_rar import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
        # from timm_vit.vit import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
        model = eval(args.model)(
            pretrained=(not args.scratch),
            img_size=args.crop, num_classes=args.num_classes,
            patch_size=args.patch, args=args
        )
    elif args.model in ['deit_tiny_patch16_224', 'deit_small_patch16_224']:
        from timm_vit.deit_rar import deit_tiny_patch16_224, deit_small_patch16_224
        # from timm_vit.deit import deit_tiny_patch16_224, deit_small_patch16_224
        model = eval(args.model)(
            pretrained=(not args.scratch),
            img_size=args.crop, num_classes=args.num_classes,
            patch_size=args.patch, args=args
        )
    elif args.model in ['swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
                        'swin_base_patch4_window7_224']:
        from timm_vit.swin_rar import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, \
            swin_base_patch4_window7_224
        # from timm_vit.swin import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, \
        #     swin_base_patch4_window7_224
        model = eval(args.model)(
            pretrained=(not args.scratch),
            img_size=args.crop, num_classes=args.num_classes,
            patch_size=args.patch, window_size=args.window_size, args=args
        )
        # logger.info(model)
        # get_named_paras(model)
    else:
        raise ValueError(args.model)
    model.train()

    # if args.mul_gpus:
    # model = nn.DataParallel(model)

    if args.train_mode == 'nat':
        train_natural(args, model, train_loader, test_loader)
    elif args.train_mode in ['fgsm', 'pgd', 'trades']:
        train_adv(args, model, train_loader, test_loader)
    else:
        raise ValueError(args.train_mode)


if __name__ == "__main__":
    main(Train_Config(get_args()))
