import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple
from timm.models.layers import DropPath, trunc_normal_

from mmdet.models.builder import BACKBONES
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint, load_state_dict)
from mmcv.cnn import (Conv2d, build_norm_layer, constant_init, normal_init, trunc_normal_init, build_activation_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmdet.utils import get_root_logger
from mmdet.models.utils import PatchEmbed, nlc_to_nchw, nchw_to_nlc


class GRN(BaseModule):
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


class ConvNeXtV2Block(BaseModule):
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


class SpatialReductionAttention(MultiheadAttention):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 batch_first=True,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 init_cfg=None):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            batch_first=batch_first,
            dropout_layer=dropout_layer,
            bias=qkv_bias,
            init_cfg=init_cfg)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape, identity=None):

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class PVTEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 use_conv_ffn=False,
                 init_cfg=None):
        super(PVTEncoderLayer, self).__init__(init_cfg=init_cfg)

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = SpatialReductionAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            use_conv=use_conv_ffn,
            act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)

        return x


@BACKBONES.register_module()
class CustomBackbone(BaseModule):
    """ ConvNeXt V2 + PVT V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,
                 in_channels=3,
                 # Specific parameters for ConvNext V2
                 convnext_depths=[2, 2, 6, 2],
                 convnext_dims=[40, 80, 160, 320],
                 convnext_drop_path_rate=0.,
                 # End Specific parameters for ConvNext V2
                 # Specific parameters for PVT V2
                 pvt_embed_dims=[32, 64, 160, 256],
                 pvt_num_stages=4,
                 pvt_num_layers=[3, 4, 6, 3],
                 pvt_num_heads=[1, 2, 5, 8],
                 pvt_patch_sizes=[7, 3, 3, 3],
                 pvt_strides=[4, 2, 2, 2],
                 pvt_paddings=[3, 1, 1, 1],
                 pvt_sr_ratios=[8, 4, 2, 1],
                 pvt_mlp_ratios=[8, 8, 4, 4],
                 pvt_qkv_bias=True,
                 pvt_drop_rate=0.,
                 pvt_attn_drop_rate=0.,
                 pvt_drop_path_rate=0.1,
                 pvt_use_conv_ffn=True,
                 pvt_act_cfg=dict(type='GELU'),
                 pvt_norm_cfg=dict(type='LN', eps=1e-6),
                 # End Specific parameters for PVT V2
                 align_directions=['T2C', 'T2C', 'T2C', 'T2C'],
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = out_indices
        self.init_cfg = init_cfg
        self.align_directions = align_directions

        # ==========================Begin ConvNext V2============================ #
        # ======================Using ConvNext V2 atto model===================== #
        self.depths = convnext_depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels, convnext_dims[0], kernel_size=4, stride=4),
            LayerNorm(convnext_dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(convnext_dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(convnext_dims[i], convnext_dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, convnext_drop_path_rate, sum(convnext_depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=convnext_dims[i], drop_path=dp_rates[cur + j]) for j in range(convnext_depths[i])]
            )
            self.stages.append(stage)
            cur += convnext_depths[i]

        convnext_norm_cfg = dict(type='LN')
        for i in range(len(out_indices)):
            layer = build_norm_layer(convnext_norm_cfg, convnext_dims[i])[1]
            layer_name = f'norm{out_indices[i]}'
            self.add_module(layer_name, layer)
        # ==========================End ConvNext V2============================ #

        # ==========================Begin PVT V2 ============================ #
        # ========================Using PVT V2 b0 model====================== #
        self.embed_dims = pvt_embed_dims
        self.num_stages = pvt_num_stages
        self.num_layers = pvt_num_layers
        self.num_heads = pvt_num_heads
        self.patch_sizes = pvt_patch_sizes
        self.strides = pvt_strides
        self.sr_ratios = pvt_sr_ratios
        assert pvt_num_stages == len(pvt_num_layers) == len(pvt_num_heads) \
               == len(pvt_patch_sizes) == len(pvt_strides) == len(pvt_sr_ratios)

        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, pvt_drop_path_rate, sum(pvt_num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(pvt_num_layers):
            embed_dims_i = pvt_embed_dims[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=pvt_patch_sizes[i],
                stride=pvt_strides[i],
                padding=pvt_paddings[i],
                bias=True,
                norm_cfg=pvt_norm_cfg)

            layers = ModuleList()
            layers.extend([
                PVTEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=pvt_num_heads[i],
                    feedforward_channels=pvt_mlp_ratios[i] * embed_dims_i,
                    drop_rate=pvt_drop_rate,
                    attn_drop_rate=pvt_attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=pvt_qkv_bias,
                    act_cfg=pvt_act_cfg,
                    norm_cfg=pvt_norm_cfg,
                    sr_ratio=pvt_sr_ratios[i],
                    use_conv_ffn=pvt_use_conv_ffn) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(pvt_norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layers, norm]))
            cur += num_layer
        # ==========================End PVT V2============================ #

        # ==========================Public function============================ #
        # Making feature align and feature fusion between ConvNext V2 and PVT V2#
        self.align_layers = ModuleList()
        self.feature_fusion_layers = ModuleList()
        for i, align_direction in enumerate(self.align_directions):
            if align_direction == 'T2C':
                align_layer = nn.Conv2d(pvt_embed_dims[i], convnext_dims[i], kernel_size=1, stride=1)
                # fusion_out_channel = convnext_dims[i]
            else:
                align_layer = nn.Conv2d(convnext_dims[i], pvt_embed_dims[i], kernel_size=1, stride=1)
                # fusion_out_channel = pvt_embed_dims[i]
            self.align_layers.append(align_layer)
            # 这里使用pointwise方法实现两者特征的融合
            # fusion_in_channel = pvt_embed_dims[i] + convnext_dims[i]
            # self.feature_fusion = torch.nn.Conv2d(fusion_in_channel, fusion_out_channel, kernel_size=1)

        # ==========================Creating normalization layer for CNN and ViT============================#
        mix_norm_cfg = dict(type='LN')
        for i in range(len(out_indices)):
            if self.align_directions[i] == 'T2C':
                layer = build_norm_layer(mix_norm_cfg, convnext_dims[i])[1]
            else:
                layer = build_norm_layer(mix_norm_cfg, pvt_embed_dims[i])[1]
            layer_name = f'mix_norm{out_indices[i]}'
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

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
                elif isinstance(m, AbsolutePositionEmbedding):
                    m.init_weights()
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        outs = []

        convnext_x = x
        pvt_x = x
        for i in range(4):
            # ConvNext V2
            convnext_x = self.downsample_layers[i](convnext_x)
            convnext_x = self.stages[i](convnext_x)  # (N, C, H, W)
            B, C, H, W = convnext_x.shape
            convnext_norm_layer = getattr(self, f'norm{i}')
            convnext_out = convnext_x.view(B, C, H * W).permute(0, 2, 1)
            convnext_out = convnext_norm_layer(convnext_out)
            convnext_out = convnext_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

            # PVT V2
            pvt_layer = self.layers[i]
            pvt_x, hw_shape = pvt_layer[0](pvt_x)
            for block in pvt_layer[1]:
                pvt_x = block(pvt_x, hw_shape)
            pvt_x = pvt_layer[2](pvt_x)
            pvt_x = nlc_to_nchw(pvt_x, hw_shape)
            pvt_out = pvt_x

            if i in self.out_indices:
                align_directoin = self.align_directions[i]
                align_layer = self.align_layers[i]
                if align_directoin == 'T2C':
                    pvt_out = align_layer(pvt_out)
                else:
                    convnext_out = align_layer(convnext_out)
                mix_out = convnext_out + pvt_out
                B, C, H, W = mix_out.shape
                mix_norm_layer = getattr(self, f'mix_norm{i}')
                mix_out = mix_out.view(B, C, H * W).permute(0, 2, 1)
                mix_out = mix_norm_layer(mix_out)
                mix_out = mix_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(mix_out)
        # for i in range(4):
        #     convnext_x = self.downsample_layers[i](convnext_x)
        #     convnext_x = self.stages[i](convnext_x)  # (N, C, H, W)
        #     print("ConvNext===={}==={}".format(i, convnext_x.shape))
        #     if i in self.out_indices:
        #         B, C, H, W = convnext_x.shape
        #         current_norm_layer = getattr(self, f'norm{i}')
        #         out = convnext_x.view(B, C, H * W).permute(0, 2, 1)
        #         out = current_norm_layer(out)
        #         out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        #         outs.append(out)

        # for i, layer in enumerate(self.layers):
        #     pvt_x, hw_shape = layer[0](pvt_x)
        #     for block in layer[1]:
        #         pvt_x = block(pvt_x, hw_shape)
        #     pvt_x = layer[2](pvt_x)
        #     pvt_x = nlc_to_nchw(pvt_x, hw_shape)
        #     print("PVT===={}==={}".format(i, pvt_x.shape))
        #     if i in self.out_indices:
        #         pvt_out = self.align_layers[i](pvt_x)
        #         outs.append(pvt_out)

        return outs


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = CustomBackbone(
        in_channels=3,
        convnext_depths=[2, 2, 6, 2],
        convnext_dims=[40, 80, 80, 160],
        convnext_drop_path_rate=0.,
        pvt_embed_dims=[16, 32, 160, 256],
        pvt_num_heads=[1, 2, 5, 8],
        pvt_num_layers=[2, 2, 2, 2],
        align_directions=['T2C', 'T2C', 'C2T', 'C2T'],
        out_indices=(0, 1, 2, 3),
        # init_cfg=None,
        init_cfg=dict(
            checkpoint='E:/08-PythonProject/20221214 EfficientViT/EfficientViTMMDetection/models/hub/checkpoints/convnextv2_atto_pvtv2_b0.pth'),
    )
    print(model)
    y = model(x)
    print(y)
