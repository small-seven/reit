""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
import pdb
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)

from timm.models.vision_transformer import resize_pos_embed, _cfg, default_cfgs

import sys

sys.path.append("..")
from utils import load_pretrained_model_from_path, get_args
# from timm_vit.helpers import load_pretrained
from config import Train_Config

args_vit = Train_Config(get_args())


class DataNorm(nn.Module):
    def __init__(self, mean, std):
        super(DataNorm, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return tensor.sub(mean).div(std)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if args_vit.r_std != 0.0:
            r = args_vit.r_std * torch.randn(B, 1, C).to(x.device)
            if args_vit.eval and not self.training:
                R = self.get_R_testing(x, r)
            else:
                R = self.compute_R(x, r, q)
        else:
            R = torch.zeros_like(x).to(x.device)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        y = x + R
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

    def get_R_testing(self, x, r1):
        B, N, C = x.shape
        r2_data = args_vit.r_std * torch.randn(B, 1, C).to(x.device)
        # x_new = torch.zeros_like(x).to(x.device)
        # x_new.data = x.data
        # r2 = args_vit.r_std * torch.randn(B, 1, C).to(x.device)
        # r2.requires_grad = True

        sim_min, r2_best = 1., r2_data
        for i in range(args_vit.r_steps):
            r2 = torch.zeros_like(r2_data).to(x.device)
            r2.data = r2_data
            r2.requires_grad = True

            x_new = torch.zeros_like(x).to(x.device)
            x_new.data = x.data
            x_new.requires_grad = True

            R1 = self.compute_R(x_new, r1)
            R2 = self.compute_R(x_new, r2)

            X_grad_R1 = torch.autograd.grad(R1.mean(), x_new, retain_graph=True, create_graph=True)
            X_grad_R2 = torch.autograd.grad(R2.mean(), x_new, retain_graph=True, create_graph=True)
            loss_sim_XgradR = torch.cosine_similarity(X_grad_R1[0], X_grad_R2[0]).mean()

            loss_sim_XgradR.backward()
            with torch.no_grad():
                r2.data -= args_vit.r_alpha * torch.sign(r2.grad.detach())
                r2_std = torch.std(r2.data, dim=-1, keepdim=True)
                r2_mean = torch.mean(r2.data, dim=-1, keepdim=True)
                r2.data = args_vit.r_std * (r2.data - r2_mean) / r2_std
                r2_data = r2.data
                r2.grad.zero_()

                if sim_min >= loss_sim_XgradR.item():
                    sim_min = loss_sim_XgradR.item()
                    r2_best = r2_data

        return random.sample([r1.data, r2_best], 1)[0]

    def compute_R(self, x, r, q=None):
        B, N, C = x.size()

        if q is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0]

        R_qkv = self.qkv(r).reshape(B, 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        Rk, Rv = R_qkv[1], R_qkv[2]

        R = ((q @ Rk.transpose(-2, -1)) @ Rv).transpose(1, 2).reshape(B, N, C)

        return R


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.data_norm = DataNorm(mean=args_vit.data_mean, std=args_vit.data_std)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        nn.init.constant_(self.head.weight, 0.)
        nn.init.constant_(self.head.bias, 0.)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            pass
            # trunc_normal_(m.weight, std=.02)
            # if isinstance(m, nn.Linear) and m.bias is not None:
            #     nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.data_norm(x)
        x = self.forward_features(x)
        x = self.head(x)
        return x


def checkpoint_filter_fn(state_dict, model, args):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():

        if args.pretrain_pos_only:
            if not k in ['pos_embed', 'head.weight', 'head.bias']:
                # head.weight and head.bias will be ignored in the timm library
                continue

        if k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        elif k == 'patch_embed.proj.weight' and v.shape != model.patch_embed.proj.weight.shape:
            # Resize kernel
            _logger.warning("Patch size doesn't match. ")

            if True:
                _logger.warning('Downsample patch embedding')
                # try:
                v = v.reshape(*v.shape[:2], args.patch, v.shape[2] // args.patch,
                              args.patch, v.shape[3] // args.patch).sum(dim=[3, 5])
                # except IndexError:
                #     v = v.reshape(768, 3, 16, 16)
                #     v = v.reshape(*v.shape[:2], args.patch, v.shape[2] // args.patch,
                #                   args.patch, v.shape[3] // args.patch).sum(dim=[3, 5])
            else:
                if args.patch_embed_scratch:
                    _logger.warning('Use initialized patch embedding')
                    continue
                elif args.downsample_factor:
                    _logger.warning('Downsample patch embedding')
                    v = v.reshape(*v.shape[:2],
                                  args.patch, v.shape[2] // args.patch,
                                  args.patch, v.shape[3] // args.patch).sum(dim=[3, 5])
                else:
                    _logger.warning('Downsample patch embedding with F.interpolate')
                    v = F.interpolate(v, model.patch_embed.proj.weight.shape[-1])
        out_dict[k] = v
    for key in model.state_dict().keys():
        if not key in out_dict:
            _logger.warning('Initialized {}'.format(key))
    return out_dict


def _create_vision_transformer(variant, pretrained=False, distilled=False, **kwargs):
    default_cfg = default_cfgs[variant]
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-1]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model_cls = DistilledVisionTransformer if distilled else VisionTransformer
    model = model_cls(img_size=img_size, num_classes=num_classes, representation_size=repr_size, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        # load_pretrained(
        #     model, num_classes=num_classes, in_chans=kwargs.get('in_chans', 3),
        #     filter_fn=partial(checkpoint_filter_fn, args=kwargs.pop('args'), model=model))
        load_pretrained_model_from_path(
            model, _logger=_logger, nci=kwargs.pop('nci', False),
            num_classes=num_classes, in_chans=kwargs.get('in_chans', 3),
            filter_fn=partial(checkpoint_filter_fn, args=kwargs.pop('args'), model=model),
            strict=False
        )
    else:
        _logger.warning('Training from scratch')
    return model


def vit_base_patch2(pretrained=False, **kwargs):
    assert not pretrained

    model = VisionTransformer(
        img_size=kwargs.pop('img_size'),
        num_classes=kwargs.pop('num_classes'),
        patch_size=2, embed_dim=768, depth=8, num_heads=8)

    return model


def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch16_224_in21k(pretrained=False, patch_size=16, args=None, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    depth = args.num_layers or 12
    _logger.info('Number of layers: {}'.format(depth))
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=768, depth=depth, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, args=args, **model_kwargs)
    return model


def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_in21k(pretrained=False, patch_size=16, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


def vit_tiny_patch16_224(pretrained=False, patch_size=16, args=None, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    depth = args.num_layers or 12
    _logger.info('Number of layers: {}'.format(depth))
    model_kwargs = dict(patch_size=patch_size, embed_dim=192, depth=depth, num_heads=3, representation_size=192,
                        **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, args=args, **model_kwargs)
    return model


def vit_small_patch16_224(pretrained=False, patch_size=16, args=None, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    depth = args.num_layers or 12
    _logger.info('Number of layers: {}'.format(depth))
    model_kwargs = dict(patch_size=patch_size, embed_dim=384, depth=depth, num_heads=6, representation_size=384,
                        **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, args=args, **model_kwargs)
    return model


def vit_base_patch16_224(pretrained=False, patch_size=16, args=None, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    depth = args.num_layers or 12
    _logger.info('Number of layers: {}'.format(depth))
    model_kwargs = dict(patch_size=patch_size, embed_dim=768, depth=depth, num_heads=12, representation_size=768,
                        **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, args=args, **model_kwargs)
    return model
