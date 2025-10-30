# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import re
from collections import OrderedDict
from typing import Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from mmcv.runner import (_load_checkpoint, load_state_dict)
from mmcv.cnn import build_norm_layer

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import nlc_to_nchw
from mmdet.utils import get_root_logger

from mmdet.models.backbones.auxiliary_framework_swin_b import FLFAuxiliaryFramework, FeatureAggregationExpansionModule


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


def convnextv2_load_checkpoint(
        model: torch.nn.Module,
        filename: str,
        map_location: Union[str, Callable, None] = None,
        strict: bool = False,
        logger: Optional[logging.Logger] = None,
        revise_keys: list = [(r'^module\.', '')]) -> Union[dict, OrderedDict]:
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


@BACKBONES.register_module()
class ConvNeXtV2Auxiliary(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,
                 in_chans=3,
                 img_size=224,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None,
                 mobilenet_pretrained=None,
                 foundation_pretrained=None
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.out_indices = out_indices
        self.init_cfg = init_cfg
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        norm_cfg = dict(type='LN')
        for i in range(len(out_indices)):
            layer = build_norm_layer(norm_cfg, dims[i])[1]
            layer_name = f'norm{out_indices[i]}'
            self.add_module(layer_name, layer)

        # 增加auxiliary framework
        source_embed_dims = dims
        self.auxiliary_net = FLFAuxiliaryFramework(img_size=img_size,
                                                   source_embed_dims=source_embed_dims,
                                                   foundation_pretrained=foundation_pretrained)
        self.fae_module = FeatureAggregationExpansionModule(source_embed_dims=source_embed_dims, fusion_layer=1, need_bn=False)

        self.apply(self._init_weights)
        self.init_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_backbone(self):
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg
            logger = get_root_logger()
            convnextv2_load_checkpoint(self, self.init_cfg['checkpoint'], map_location='cpu', strict=False,
                                       logger=logger)

    def forward(self, x):
        # Begin Step 1: get features from foundation model and mobilenet
        vit_features = self.auxiliary_net(x)
        # End Step 1: get features from foundation model and mobilenet

        # Step 2: fusion featuers from foundation model and mobilenet
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)

            x = self.stages[i](x)  # (N, C, H, W)
            if i in self.out_indices:
                # B, C, H, W = x.shape
                # current_norm_layer = getattr(self, f'norm{i}')
                # out = x.view(B, C, H * W).permute(0, 2, 1)
                # out = current_norm_layer(out)
                # out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

                # Begin Step 2.2 fuse feature from foundation model
                vit_feature = vit_features[i]
                vit_b_adapter = getattr(self.auxiliary_net, "vit_b_adapter_{}".format(i))
                vit_b_fusion = getattr(self.auxiliary_net, "vit_b_fusion_{}".format(i))
                vit_feature_adapter = vit_b_adapter(vit_feature)
                x = vit_b_fusion(x, vit_feature_adapter)
                # End Step 2.2 fuse feature from foundation model

                outs.append(x)

        # Begin Step 3: fuse and expand the output features
        outs = self.fae_module(outs)
        # End Step 3: fuse and expand the output features

        return outs


if __name__ == '__main__':
    x = torch.rand(1, 3, 448, 448)
    model = ConvNeXtV2Auxiliary(
        in_chans=3,
        img_size=448,
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        drop_path_rate=0.,
        init_cfg=dict(
            checkpoint='E:/08-PythonProject/20221214 EfficientViT/EfficientViTMMDetection/models/hub/checkpoints/convnextv2_atto_1k_224_ema.pt'),
        
        foundation_pretrained='E:/08-PythonProject/20221214 EfficientViT/EfficientViTMMDetection/models/hub/checkpoints/auxiliary_vit-b-win.pth'
    )
    print(model)
    y = model(x)
    print(y)
