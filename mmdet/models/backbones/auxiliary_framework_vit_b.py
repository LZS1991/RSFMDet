import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple
from timm.models.layers import DropPath, trunc_normal_

from mmcv.runner import (BaseModule, Sequential, _load_checkpoint, load_state_dict)
from mmcv.cnn import (Conv2d, build_norm_layer, ConvModule, build_activation_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.utils import get_root_logger
from mmdet.models.utils import nlc_to_nchw, nchw_to_nlc

# 从遥感大模型中引入特征处理模块，为本算法的模块提供特征
from mmdet.models.backbones.vit_win_rvsa_wsz7 import ViT_Win_RVSA_V3_WSZ7, Norm2d


class AbsolutePositionEmbedding(BaseModule):
    def __init__(self, pos_shape, pos_dim, drop_rate=0., init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(pos_shape, int):
            pos_shape = to_2tuple(pos_shape)
        elif isinstance(pos_shape, tuple):
            if len(pos_shape) == 1:
                pos_shape = to_2tuple(pos_shape[0])
            assert len(pos_shape) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pos_shape)}'
        self.pos_shape = pos_shape
        self.pos_dim = pos_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_shape[0] * pos_shape[1], pos_dim))
        self.drop = nn.Dropout(p=drop_rate)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

    def resize_pos_embed(self, pos_embed, input_shape, mode='bilinear'):
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = self.pos_shape
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, self.pos_dim).permute(0, 3, 1, 2).contiguous()
        pos_embed_weight = F.interpolate(
            pos_embed_weight, size=input_shape, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight,
                                         2).transpose(1, 2).contiguous()
        pos_embed = pos_embed_weight

        return pos_embed

    def forward(self, x, hw_shape, mode='bilinear'):
        pos_embed = self.resize_pos_embed(self.pos_embed, hw_shape, mode)
        return self.drop(x + pos_embed)


class LayerNorm(BaseModule):
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


class MixFFN(BaseModule):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 use_conv=False,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        if use_conv:
            # 3x3 depth wise conv to provide positional encode information
            dw_conv = Conv2d(
                in_channels=feedforward_channels,
                out_channels=feedforward_channels,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
                bias=True,
                groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, activate, drop, fc2, drop]
        if use_conv:
            layers.insert(1, dw_conv)
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class HydraAttention(BaseModule):
    def __init__(self,
                 embed_dims=1
                 ):
        super().__init__()

        self.out_proj = nn.Linear(embed_dims, embed_dims, bias=True)

    def forward(self, x_q, x_kv, identity=None, hw_shape=None):
        if identity is None:
            identity = x_q

        q = x_q / x_q.norm(dim=-1, keepdim=True)
        k = x_kv / x_kv.norm(dim=-1, keepdim=True)
        kv = (k * x_kv).sum(dim=-2, keepdim=True)
        out = q * kv

        out = identity + out

        return out


class BridgeLayer(BaseModule):
    """Bridging Modele in BAN.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels=4,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super().__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = HydraAttention(embed_dims=embed_dims)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, x_kv):
        hw_shape = x.shape[-2:]
        x = nchw_to_nlc(x)
        x_kv = nchw_to_nlc(x_kv)
        x = self.attn(self.norm1(x), x_kv, identity=x, hw_shape=hw_shape)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        x = x + x_kv
        x = nlc_to_nchw(x, hw_shape)
        return x


class FLFAuxiliaryFramework(BaseModule):
    # Auxiliary Framework with Fusion of Foundation and Lightweight Model
    def __init__(self,
                 in_channels=3,
                 img_size=224,
                 # Specific parameters for detection Backbone
                 source_embed_dims=[32, 64, 160, 256],
                 out_indices=(0, 1, 2, 3),
                 init_cfg=[
                     dict(type='Kaiming',
                          layer=['Conv2d'],
                          nonlinearity='leaky_relu'),
                     dict(type='Normal', layer=['Linear'], std=0.01),
                     dict(type='Constant', layer=['BatchNorm2d'], val=1)
                 ],
                 foundation_pretrained=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = out_indices
        self.init_cfg = init_cfg
        self.img_size = img_size
        self.source_embed_dims = source_embed_dims
        self.foundation_pretrained = foundation_pretrained

        # ==========================Begin FoundationModel=========================== #
        vit_b_embed_dim = 768
        self.vit_b = ViT_Win_RVSA_V3_WSZ7(
            img_size=img_size, embed_dim=vit_b_embed_dim, pretrained=foundation_pretrained)
        self.vit_b_drop = DropPath(0.1)
        #  freeze vit_b parameters
        for param in self.vit_b.parameters():
            param.requires_grad = False
        # ==========================End FoundationModel=========================== #

        # ===================Begin Adapter of FoundationModel===================== #
        for i, source_embed_dim in enumerate(self.source_embed_dims):
            vit_b_adapter = nn.Sequential(
                nn.ConvTranspose2d(vit_b_embed_dim, source_embed_dim, kernel_size=1, stride=1, padding=0),
                Norm2d(source_embed_dim),
                nn.GELU(),
            )
            vit_b_adapter_layer_name = 'vit_b_adapter_{}'.format(i)
            self.add_module(vit_b_adapter_layer_name, vit_b_adapter)

            vit_b_fusion = BridgeLayer(embed_dims=source_embed_dim)
            vit_b_fusion_layer_name = 'vit_b_fusion_{}'.format(i)
            self.add_module(vit_b_fusion_layer_name, vit_b_fusion)
        # ===================End Adapter of FoundationModel===================== #

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        logger = get_root_logger()
        self.apply(self._init_weights)

    def forward(self, x):
        # extract remote sensing features from image by foundation model
        vit_features = self.vit_b(x)
        return vit_features


# Expansion Module for  Feature Aggregatio Expansion
class FeatureAggregationExpansionModule(BaseModule):
    def __init__(self,
                 source_embed_dims=[32, 64, 160, 256],
                 out_indices=(0, 1, 2, 3),
                 drop_path_rate=0.1,
                 need_bn=False,
                 fusion_layer=0,
                 init_cfg=[
                     dict(type='Kaiming',
                          layer=['Conv2d'],
                          nonlinearity='leaky_relu'),
                     dict(type='Normal', layer=['Linear'], std=0.01),
                     dict(type='Constant', layer=['BatchNorm2d'], val=1)
                 ]
                 ):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = out_indices
        self.init_cfg = init_cfg
        self.source_embed_dims = source_embed_dims
        self.need_bn = need_bn
        self.fusion_layer = fusion_layer

        # ====================Begin of Finally Processing====================== #
        inter_channel = source_embed_dims[0] + source_embed_dims[1] + source_embed_dims[2] + source_embed_dims[3]
        # 融合特征中获取p2特征
        self.inter_to_p2 = nn.Sequential(
            nn.Conv2d(inter_channel, source_embed_dims[0], kernel_size=1, stride=1, padding=0),
            build_norm_layer(dict(type='BN', eps=1e-6), source_embed_dims[0])[1],
            nn.ReLU(),
        )
        self.inter_to_p2_drop = DropPath(0.1)

        # 融合特征中获取p3特征
        self.inter_to_p3 = nn.Sequential(
            nn.Conv2d(inter_channel, source_embed_dims[1], kernel_size=1, stride=1, padding=0),
            build_norm_layer(dict(type='BN', eps=1e-6), source_embed_dims[1])[1],
            nn.ReLU(),
        )
        self.inter_to_p3_drop = DropPath(0.1)

        # 融合特征中获取p4特征
        self.inter_to_p4 = nn.Sequential(
            nn.Conv2d(inter_channel, source_embed_dims[2], kernel_size=1, stride=1, padding=0),
            build_norm_layer(dict(type='BN', eps=1e-6), source_embed_dims[2])[1],
            nn.ReLU(),
        )
        self.inter_to_p4_drop = DropPath(0.1)

        # 融合特征中获取p5特征
        self.inter_to_p5 = nn.Sequential(
            nn.Conv2d(inter_channel, source_embed_dims[3], kernel_size=1, stride=1, padding=0),
            build_norm_layer(dict(type='BN', eps=1e-6), source_embed_dims[3])[1],
            nn.ReLU(),
        )
        self.inter_to_p5_drop = DropPath(0.1)

        self.source_up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.source_up_sample_4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.source_up_sample_8 = nn.Upsample(scale_factor=8, mode='nearest')
        self.source_down_sample_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.source_down_sample_4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.source_down_sample_8 = nn.AvgPool2d(kernel_size=8, stride=8)

        self.inter_up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.inter_up_sample_4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.inter_up_sample_8 = nn.Upsample(scale_factor=8, mode='nearest')
        self.inter_down_sample_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.inter_down_sample_4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.inter_down_sample_8 = nn.AvgPool2d(kernel_size=8, stride=8)
        # =====================End of Finally Processing======================= #

        adapter_dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, 4)
        ]  # stochastic num_layer decay rule

        self.adapter_dropout_0 = DropPath(adapter_dpr[1])
        self.adapter_dropout_1 = DropPath(adapter_dpr[1])
        self.adapter_dropout_2 = DropPath(adapter_dpr[2])
        self.adapter_dropout_3 = DropPath(adapter_dpr[3])

        # new
        if self.need_bn:
            norm_cfg = dict(type='LN')
            for i in range(len(out_indices)):
                layer = build_norm_layer(norm_cfg, source_embed_dims[i])[1]
                layer_name = f'norm{out_indices[i]}'
                self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, source_outs):
        # 即将输出的特征均存储在outs中，其中第一层特征其实是不用的，因此只对后面的3个特征进行处理即可，对其进行融合和分割
        out_p2 = source_outs[0]
        out_p3 = source_outs[1]
        out_p4 = source_outs[2]
        out_p5 = source_outs[3]

        if self.fusion_layer == 0:
            out_p2_inter = out_p2
            out_p3_inter = self.source_up_sample_2(out_p3)
            out_p4_inter = self.source_up_sample_4(out_p4)
            out_p5_inter = self.source_up_sample_8(out_p5)
        elif self.fusion_layer == 1:
            out_p2_inter = self.source_down_sample_2(out_p2)
            out_p3_inter = out_p3
            out_p4_inter = self.source_up_sample_2(out_p4)
            out_p5_inter = self.source_up_sample_4(out_p5)
        elif self.fusion_layer == 2:
            out_p2_inter = self.source_down_sample_4(out_p2)
            out_p3_inter = self.source_down_sample_2(out_p3)
            out_p4_inter = out_p4
            out_p5_inter = self.source_up_sample_2(out_p5)
        else:
            out_p2_inter = self.source_down_sample_8(out_p2)
            out_p3_inter = self.source_down_sample_4(out_p3)
            out_p4_inter = self.source_down_sample_2(out_p4)
            out_p5_inter = out_p5

        out_inter = torch.cat([out_p2_inter, out_p3_inter], dim=1)
        out_inter = torch.cat([out_inter, out_p4_inter], dim=1)
        out_inter = torch.cat([out_inter, out_p5_inter], dim=1)  # 此时，out_inter的C=三者通道之和

        out_p2_refine = self.inter_to_p2(out_inter)
        out_p3_refine = self.inter_to_p3(out_inter)
        out_p4_refine = self.inter_to_p4(out_inter)
        out_p5_refine = self.inter_to_p5(out_inter)

        if self.fusion_layer == 0:
            out_p3_refine = self.inter_down_sample_2(out_p3_refine)
            out_p4_refine = self.inter_down_sample_4(out_p4_refine)
            out_p5_refine = self.inter_down_sample_8(out_p5_refine)
        elif self.fusion_layer == 1:
            out_p2_refine = self.inter_up_sample_2(out_p2_refine)
            out_p4_refine = self.inter_down_sample_2(out_p4_refine)
            out_p5_refine = self.inter_down_sample_4(out_p5_refine)
        elif self.fusion_layer == 2:
            out_p2_refine = self.inter_up_sample_4(out_p2_refine)
            out_p3_refine = self.inter_up_sample_2(out_p3_refine)
            out_p5_refine = self.inter_down_sample_2(out_p5_refine)
        else:
            out_p2_refine = self.inter_up_sample_8(out_p2_refine)
            out_p3_refine = self.inter_up_sample_4(out_p3_refine)
            out_p4_refine = self.inter_up_sample_2(out_p4_refine)

        out_p2 = out_p2 + self.adapter_dropout_0(out_p2_refine)
        out_p3 = out_p3 + self.adapter_dropout_1(out_p3_refine)
        out_p4 = out_p4 + self.adapter_dropout_2(out_p4_refine)
        out_p5 = out_p5 + self.adapter_dropout_3(out_p5_refine)

        outs = []
        outs.append(out_p2)
        outs.append(out_p3)
        outs.append(out_p4)
        outs.append(out_p5)

        # new
        if self.need_bn:
            for idx, out in enumerate(outs):
                B, C, H, W = out.shape
                current_norm_layer = getattr(self, f'norm{idx}')
                out = out.view(B, C, H * W).permute(0, 2, 1)
                out = current_norm_layer(out)
                out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs[idx] = out

        return outs
