'''
Lite Vision Transformer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from mmdet.models.builder import BACKBONES
from mmdet.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, pvt_convert

from mmdet.models.backbones.auxiliary_framework_swin_b import FLFAuxiliaryFramework, FeatureAggregationExpansionModule


def nlc_to_nhwc(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.reshape(B, H, W, C).contiguous()


def nhwc_to_nlc(x):
    """Flatten a 4D tensor of shape [bs, c, h, w] to a 2D tensor of shape [bs, (c*h*w)].

    Args:
        x (Tensor): The input tensor of shape [bs, c, h, w].

    Returns:
        Tensor: The output tensor of shape [bs, (c*h*w)].
    """
    B, H, W, C = x.shape
    return x.view(B, H * W, C)


class ds_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 dilation=[1, 3, 5], groups=1, bias=True,
                 act_layer='nn.SiLU(True)', init='kaiming'):
        super().__init__()
        assert in_planes % groups == 0
        assert kernel_size == 3, 'support kernel size 3 now'
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
        self.act = eval(act_layer)
        self.init = init
        self._initialize_weights()

    def _initialize_weights(self):
        if self.init == 'dirac':
            nn.init.dirac_(self.weight, self.groups)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight)
        else:
            raise NotImplementedError
        if self.with_bias:
            if self.init == 'dirac':
                nn.init.constant_(self.bias, 0.)
            elif self.init == 'kaiming':
                bound = self.groups / (self.kernel_size ** 2 * self.in_planes)
                bound = math.sqrt(bound)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                raise NotImplementedError

    def forward(self, x):
        output = 0
        for dil in self.dilation:
            output += self.act(
                F.conv2d(
                    x, weight=self.weight, bias=self.bias, stride=self.stride, padding=dil,
                    dilation=dil, groups=self.groups,
                )
            )
        return output


class CSA(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, kernel_size=3, padding=1, stride=2,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim ** -0.5

        self.attn = nn.Linear(in_dim, kernel_size ** 4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.csa_group = 1
        assert out_dim % self.csa_group == 0
        self.weight = nn.Conv2d(
            self.kernel_size * self.kernel_size * out_dim,
            self.kernel_size * self.kernel_size * out_dim,
            1,
            stride=1, padding=0, dilation=1,
            groups=self.kernel_size * self.kernel_size * self.csa_group,
            bias=qkv_bias,
        )
        assert qkv_bias == False
        fan_out = self.kernel_size * self.kernel_size * self.out_dim
        fan_out //= self.csa_group
        self.weight.weight.data.normal_(0, math.sqrt(2.0 / fan_out))  # init

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, v=None):
        B, H, W, _ = x.shape
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
               self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = x.permute(0, 3, 1, 2)  # B,C,H, W
        v = self.unfold(v).reshape(
            B, self.out_dim, self.kernel_size * self.kernel_size, h * w
        ).permute(0, 3, 2, 1).reshape(B * h * w, self.kernel_size * self.kernel_size * self.out_dim, 1, 1)
        v = self.weight(v)
        v = v.reshape(B, h * w, self.kernel_size * self.kernel_size, self.num_heads,
                      self.out_dim // self.num_heads).permute(0, 3, 1, 2, 4).contiguous()  # B,H,N,kxk,C/H

        x = (attn @ v).permute(0, 1, 4, 3, 2)
        x = x.reshape(B, self.out_dim * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0., with_depconv=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.with_depconv = with_depconv

        if self.with_depconv:
            self.fc1 = nn.Conv2d(
                in_features, hidden_features, 1, stride=1, padding=0, dilation=1,
                groups=1, bias=True,
            )
            self.depconv = nn.Conv2d(
                hidden_features, hidden_features, 3, stride=1, padding=1, dilation=1,
                groups=hidden_features, bias=True,
            )
            self.act = act_layer()
            self.fc2 = nn.Conv2d(
                hidden_features, out_features, 1, stride=1, padding=0, dilation=1,
                groups=1, bias=True,
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.with_depconv:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.fc1(x)
            x = self.depconv(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            return x
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x


class Attention(nn.Module):
    def __init__(
            self,
            dim, num_heads=8, qkv_bias=False,
            qk_scale=None, attn_drop=0.,
            proj_drop=0.,
            rasa_cfg=None, sr_ratio=1,
            linear=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.rasa_cfg = rasa_cfg
        self.use_rasa = rasa_cfg is not None
        self.sr_ratio = sr_ratio

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        if self.use_rasa:
            if self.rasa_cfg.atrous_rates is not None:
                self.ds = ds_conv2d(
                    dim, dim, kernel_size=3, stride=1,
                    dilation=self.rasa_cfg.atrous_rates, groups=dim, bias=qkv_bias,
                    act_layer=self.rasa_cfg.act_layer, init=self.rasa_cfg.init,
                )
            if self.rasa_cfg.r_num > 1:
                self.silu = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _inner_attention(self, x):
        B, H, W, C = x.shape
        q = self.q(x).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.use_rasa:
            if self.rasa_cfg.atrous_rates is not None:
                q = q.permute(0, 1, 3, 2).reshape(B, self.dim, H, W).contiguous()
                q = self.ds(q)
                q = q.reshape(B, self.num_heads, self.dim // self.num_heads, H * W).permute(0, 1, 3, 2).contiguous()

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 3, 1, 2)
                x_ = self.sr(x_).permute(0, 2, 3, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            raise NotImplementedError

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x):
        if self.use_rasa:
            x_in = x
            x = self._inner_attention(x)
            if self.rasa_cfg.r_num > 1:
                x = self.silu(x)
            for _ in range(self.rasa_cfg.r_num - 1):
                x = x + x_in
                x_in = x
                x = self._inner_attention(x)
                x = self.silu(x)
        else:
            x = self._inner_attention(x)
        return x


class Transformer_block(nn.Module):
    def __init__(self, dim,
                 num_heads=1, mlp_ratio=3., attn_drop=0.,
                 drop_path=0., sa_layer='sa', rasa_cfg=None, sr_ratio=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=None, with_depconv=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sa_layer == 'csa':
            self.attn = CSA(
                dim, dim, num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop)
        elif sa_layer in ['rasa', 'sa']:
            self.attn = Attention(
                dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, rasa_cfg=rasa_cfg, sr_ratio=sr_ratio)
        else:
            raise NotImplementedError
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            with_depconv=with_depconv)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if self.patch_size[0] == 7:
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
        else:
            x = x.permute(0, 3, 1, 2)
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
        return x


class lite_vision_transformer(nn.Module):

    def __init__(self,
                 layers,
                 in_chans=3,
                 img_size=224,
                 patch_size=4,
                 embed_dims=None,
                 num_heads=None,
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'],
                 rasa_cfg=None,
                 mlp_ratios=None,
                 mlp_depconv=None,
                 sr_ratios=[1, 1, 1, 1],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_cfg=None,
                 mobilenet_pretrained=None,
                 foundation_pretrained=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_depconv = mlp_depconv
        self.sr_ratios = sr_ratios
        self.layers = layers
        # self.num_classes = num_classes
        self.sa_layers = sa_layers
        self.rasa_cfg = rasa_cfg
        self.init_cfg = init_cfg

        network = []
        for stage_idx in range(len(layers)):
            _patch_embed = OverlapPatchEmbed(
                patch_size=7 if stage_idx == 0 else 3,
                stride=4 if stage_idx == 0 else 2,
                in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
                embed_dim=embed_dims[0] if stage_idx == 0 else embed_dims[stage_idx],
            )

            _blocks = []
            for block_idx in range(layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
                _blocks.append(Transformer_block(
                    embed_dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    mlp_ratio=mlp_ratios[stage_idx],
                    sa_layer=sa_layers[stage_idx],
                    rasa_cfg=self.rasa_cfg if sa_layers[stage_idx] == 'rasa' else None,  # I am here
                    sr_ratio=sr_ratios[stage_idx],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop_rate, drop_path=block_dpr,
                    with_depconv=mlp_depconv[stage_idx]))
            _blocks = nn.Sequential(*_blocks)

            network.append(nn.Sequential(
                _patch_embed,
                _blocks
            ))

        # backbone
        self.backbone = nn.ModuleList(network)

        self.downstream_norms = nn.ModuleList([nn.Identity()
                                               for idx in range(len(embed_dims))])
        self.apply(self._init_weights)
        self.init_backbone()

        # 增加auxiliary framework
        source_embed_dims = embed_dims
        self.auxiliary_net = FLFAuxiliaryFramework(img_size=img_size,
                                                   source_embed_dims=source_embed_dims,
                                                   foundation_pretrained=foundation_pretrained)
        self.mobilenet_dims = []

        norm_cfg = dict(type='LN')
        for i in range(len(layers)):
            layer = build_norm_layer(norm_cfg, embed_dims[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

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
            pretrained = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(pretrained, strict=True)

    def forward(self, x):
        # Begin Step 1: get features from foundation model and mobilenet
        vit_features = self.auxiliary_net(x)
        # End Step 1: get features from foundation model and mobilenet

        # Step 2: fusion featuers from foundation model and mobilenet
        outs = []
        for idx, stage in enumerate(self.backbone):
            x = stage(x)

            # Begin Step 2.2 fuse feature from foundation model
            vit_feature = vit_features[idx]
            vit_b_adapter = getattr(self.auxiliary_net, "vit_b_adapter_{}".format(idx))
            vit_b_fusion = getattr(self.auxiliary_net, "vit_b_fusion_{}".format(idx))
            vit_feature_adapter = vit_b_adapter(vit_feature)
            _,H, W, _ = x.shape
            current_norm_layer = getattr(self, f'norm{idx}')
            x = nlc_to_nchw(current_norm_layer(nhwc_to_nlc(x)), [H, W])
            x = vit_b_fusion(x, vit_feature_adapter)
            x = nlc_to_nhwc(nchw_to_nlc(x), [H, W])
            # End Step 2.2 fuse feature from foundation model

            x = self.downstream_norms[idx](x)
            outs.append(x.permute(0, 3, 1, 2).contiguous())

        # Begin Step 3: fuse and expand the output features
        # outs = self.fae_module(outs)
        # End Step 3: fuse and expand the output features

        return outs


@BACKBONES.register_module()
class LiteViTAuxiliary(lite_vision_transformer):
    def __init__(self,
                 img_size=224,
                 rasa_cfg=None,
                 init_cfg=None,
                 mobilenet_pretrained=None,
                 foundation_pretrained=None):
        super().__init__(
            img_size=img_size,
            layers=[2, 2, 2, 2],
            patch_size=4,
            embed_dims=[64, 64, 160, 256],
            num_heads=[2, 2, 5, 8],
            mlp_ratios=[4, 8, 4, 4],
            mlp_depconv=[False, True, True, True],
            sr_ratios=[8, 4, 2, 1],
            sa_layers=['csa', 'rasa', 'rasa', 'rasa'],
            rasa_cfg=rasa_cfg,
            init_cfg=init_cfg,
            foundation_pretrained=foundation_pretrained
        )


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = LiteViTAuxiliary(
        img_size=224,
        # rasa_cfg=dict(
        #     atrous_rates=[1, 3, 5],  # None, [1,3,5]
        #     act_layer='nn.SiLU(True)',
        #     init='kaiming',
        #     r_num=2),  # rasa setting
        # init_cfg=dict(
        #     checkpoint='E:/08-PythonProject/20221214 EfficientViT/EfficientViTMMDetection/models/hub/checkpoints/lvt_imagenet.pth.tar'),
        
        foundation_pretrained='E:/08-PythonProject/20221214 EfficientViT/EfficientViTMMDetection/models/hub/checkpoints/auxiliary_vit-b-win.pth'
    )
    print(model)
    y = model(x)
    print(y)
