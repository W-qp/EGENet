# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath


class ConvNext_v2(nn.Module):
    def __init__(self,
                 n_classes,
                 backbone_type='A',
                 fpn_chans=256,
                 fpn_norm_layer='BN',
                 fpn_act_layer=nn.ReLU,
                 aux=True):
        super().__init__()
        self.n_classes = n_classes
        self.aux = aux

        if backbone_type == 'A':
            self.build_backbone = convnextv2_atto()
            embed_dim = self.build_backbone.dim
            depths = self.build_backbone.depths
        elif backbone_type == 'F':
            self.build_backbone = convnextv2_femto()
            embed_dim = self.build_backbone.dim
            depths = self.build_backbone.depths
        else:
            raise NotImplementedError()

        self.depths = depths
        self.num_layers = len(depths)

        self.build_neck = Build_neck(
            in_chans=embed_dim,
            out_chans=fpn_chans,
            depth=len(depths),
            norm_layer=fpn_norm_layer,
            act_layer=fpn_act_layer)

        self.build_decode_head = nn.Conv2d(fpn_chans, n_classes, kernel_size=1)

        if self.aux:
            self.aux_out = nn.Sequential(
                nn.Conv2d(embed_dim * 2 ** (len(depths) - 2), fpn_chans // 2, kernel_size=3, padding=1, bias=False),
                creat_norm_layer(fpn_norm_layer, fpn_chans // 2),
                fpn_act_layer(inplace=True) if fpn_act_layer != nn.GELU else fpn_act_layer(),
                nn.Conv2d(fpn_chans // 2, n_classes, kernel_size=1)
            )

    def forward(self, x):
        x = x if isinstance(x, tuple and list) else tuple([x])
        x = torch.cat(x, dim=1)
        _, _, H, W = x.shape
        x = self.build_backbone(x)
        aux_x = x[-2] if self.aux else None
        x = self.build_neck(x)
        x = self.build_decode_head(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        aux = self.aux if self.training else False
        if aux:
            aux_x = self.aux_out(aux_x)
            aux_x = F.interpolate(aux_x, scale_factor=2 ** (len(self.depths)), mode='bilinear', align_corners=True)
            return x, aux_x
        else:
            return x


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.dim = dims[0]
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return tuple(outs)


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


def creat_norm_layer(norm_layer, channel):
    if norm_layer == 'LN':
        norm = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(channel),
            Rearrange('b h w c -> b c h w')
        )
    elif norm_layer == 'BN':
        norm = nn.BatchNorm2d(channel)
    else:
        raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")
    return norm


class PPM(nn.Module):
    def __init__(self, ppm_in_chans, out_chans=512, pool_sizes=(1, 2, 3, 6), norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        self.pool_projs = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveMaxPool2d(pool_size),
                nn.Conv2d(ppm_in_chans, out_chans, kernel_size=1, bias=False),
                creat_norm_layer(norm_layer, out_chans),
                act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
            )for pool_size in pool_sizes)

        self.bottom = nn.Sequential(
            nn.Conv2d(ppm_in_chans + len(pool_sizes) * out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            creat_norm_layer(norm_layer, out_chans),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        )

    def forward(self, x):
        xs = [x]
        for pool_proj in self.pool_projs:
            pool_x = F.interpolate(pool_proj(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            xs.append(pool_x)

        x = torch.cat(xs, dim=1)
        x = self.bottom(x)

        return x


class FPN_neck(nn.Module):
    def __init__(self, in_chans, depth, out_chans=512, norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        self.depth = depth
        stage = [i for i in range(depth)]

        self.conv_ = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_chans * 2 ** stage[::-1][i + 1], out_chans, kernel_size=1, bias=False),
                creat_norm_layer(norm_layer, out_chans),
                act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
            )for i in range(depth - 1))

        self.fpn_conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                creat_norm_layer(norm_layer, out_chans),
                act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
            )for _ in range(depth - 1))

        self.out = nn.Sequential(
            nn.Conv2d(out_chans * depth, out_chans, kernel_size=3, padding=1, bias=False),
            creat_norm_layer(norm_layer, out_chans),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        )

    def forward(self, x):
        fpn_x = x[0]
        out = [fpn_x]
        for i in range(self.depth - 1):
            fpn_x = F.interpolate(x[i], scale_factor=2, mode='bilinear', align_corners=True)
            fpn_x = self.fpn_conv[i](fpn_x) + self.conv_[i](x[i + 1])
            x[i + 1] = fpn_x
            out.append(fpn_x)
        out = out[::-1]
        _, _, H, W = out[0].shape
        for i in range(1, len(out)):
            out[i] = F.interpolate(out[i], size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat(out, dim=1)

        return self.out(x)


class Build_neck(nn.Module):
    def __init__(self, in_chans, out_chans, depth, pool_sizes=(1, 2, 3, 6), norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        self.ppm_head = PPM(ppm_in_chans=in_chans * 2 ** (depth - 1),
                            out_chans=out_chans,
                            pool_sizes=pool_sizes,
                            norm_layer=norm_layer,
                            act_layer=act_layer)
        self.fpn_neck = FPN_neck(in_chans=in_chans,
                                 out_chans=out_chans,
                                 depth=depth,
                                 norm_layer=norm_layer,
                                 act_layer=act_layer)

    def forward(self, x):
        x = list(x)[::-1]
        x[0] = self.ppm_head(x[0])
        x = self.fpn_neck(x)

        return x


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

img = torch.ones([2, 3, 512, 512]).cuda()
net = ConvNext_v2(2).cuda()
pred = net(img)