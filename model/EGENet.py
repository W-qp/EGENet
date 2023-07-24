import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_, to_2tuple


class EGENet(nn.Module):
    """
    Build EGENet model

    Args:
        n_classes (int): Number of probabilities you want to get per pixel.
        in_chans (int): Number of input image channels. Default: 3.
        basic_dim (int): Number of basic channels.
        decoder_chans (int | None): Number of channels of decoder.
        ratio (float): Ratio of ffd hidden dim to basic dim.
        n_groups (tuple[int]): Number of groups of each stage.
        decoder_heads (int | None): Number of heads of decoder.
        window_size (int): Window size.
        depths (tuple[int]): Depths of each stage.
        drop (float): Dropout rate.
        patch_size (int | tuple(int)): Patch size. Default: 1
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        aux (bool): Whether to use auxiliary classification head. Default: True
    """
    def __init__(self,
                 n_classes,
                 in_chans=3,
                 basic_dim=64,
                 decoder_chans=None,
                 ratio=0.5,
                 n_groups=(2, 4, 8, 16),
                 decoder_heads=None,
                 window_size=8,
                 depths=(2, 2, 8, 2),
                 drop=0.1,
                 patch_size=1,
                 qkv_bias=True,
                 aux=True):
        super(EGENet, self).__init__()
        self.n_classes = n_classes
        self.use_stem = False if patch_size == 1 else True
        self.aux = aux
        decoder_heads = n_groups[:3] if decoder_heads is None else decoder_heads
        decoder_heads = (decoder_heads, decoder_heads, decoder_heads) if isinstance(decoder_heads, int) else decoder_heads
        decoder_chans = basic_dim if decoder_chans is None else decoder_chans

        if self.use_stem:
            self.stem = Stem(patch_size, in_chans, basic_dim)
            in_chans = basic_dim

        dims = [basic_dim * 2 ** i for i in range(len(depths))]
        self.x1_1 = Basic_layer(in_chans, dims[0], ratio, n_groups[0], depths[0], drop)
        self.x2_1 = Down(dims[0], dims[1], ratio, n_groups[1], depths[1], drop)
        self.x3_1 = Down(dims[1], dims[2], ratio, n_groups[2], depths[2], drop)
        self.x4_1 = Down(dims[2], dims[3], ratio, n_groups[3], depths[3], drop)
        self.x4_2 = PPM(dims[3], decoder_chans)

        self.bns = nn.ModuleList([nn.BatchNorm2d(dims[0] * 2 ** i) for i in range(4)])

        if aux:
            self.aux_out = nn.Sequential(
                nn.Conv2d(basic_dim * 2 ** (len(depths) - 2), basic_dim * 2 ** (len(depths) - 3), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(basic_dim * 2 ** (len(depths) - 3)),
                nn.ReLU(True),
                nn.Conv2d(basic_dim * 2 ** (len(depths) - 3), n_classes, kernel_size=1)
            )

        encoder_channels = [basic_dim * 2 ** i for i in range(len(depths))]
        self.decoder = Decoder(decoder_heads, encoder_channels, decoder_chans, qkv_bias, drop, window_size, n_classes)

    def forward(self, x):
        x = x if isinstance(x, tuple) else tuple(x)
        x = torch.cat(x, dim=1)
        _, _, h, w = x.shape
        x = self.stem(x) if self.use_stem else x

        x1 = self.x1_1(x)
        x2 = self.x2_1(x1)
        x3 = self.x3_1(x2)
        x4 = self.x4_1(x3)
        aux_x = x3
        x1, x2, x3, x4 = self.bns[0](x1), self.bns[1](x2), self.bns[2](x3), self.bns[3](x4)
        x4 = self.x4_2(x4)
        x = self.decoder(x1, x2, x3, x4)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) if self.use_stem else x

        aux = self.aux if self.training else False
        if aux:
            aux_x = self.aux_out(aux_x)
            aux_x = F.interpolate(aux_x, size=(h, w), mode='bilinear', align_corners=True)
            return x, aux_x
        else:
            return x


class Decoder(nn.Module):
    def __init__(self,
                 n_heads,
                 encoder_channels=(64, 128, 256, 512),
                 decoder_channels=64,
                 qkv_bias=False,
                 dropout=0.1,
                 window_size=8,
                 num_classes=1,
                 chan_ratio=8):
        super(Decoder, self).__init__()

        self.b4 = Block(dim=decoder_channels, num_heads=n_heads[-1], window_size=window_size,
                        qkv_bias=qkv_bias, drop=dropout, drop_path=dropout)

        self.b3 = Block(dim=decoder_channels, num_heads=n_heads[-2], window_size=window_size,
                        qkv_bias=qkv_bias, drop=dropout, drop_path=dropout)
        self.p3 = Fuse(encoder_channels[-2], decoder_channels)

        self.b2 = Block(dim=decoder_channels, num_heads=n_heads[-3], window_size=window_size,
                        qkv_bias=qkv_bias, drop=dropout, drop_path=dropout)
        self.p2 = Fuse(encoder_channels[-3], decoder_channels)

        self.p1 = Decoder_Head(encoder_channels[-4], decoder_channels, chan_ratio)

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels),
            Conv(decoder_channels, num_classes, kernel_size=1)
        )

    def forward(self, x1, x2, x3, x4):
        x = self.b4(x4)
        x = self.p3(x, x3)
        x = self.b3(x)
        x = self.p2(x, x2)
        x = self.b2(x)
        x = self.p1(x, x1)
        x = self.segmentation_head(x)

        return x


class Basic_layer(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, n_groups, depth=2, drop=0.):
        super(Basic_layer, self).__init__()
        dim = int(out_channels * ratio)
        self.n = depth
        for i in range(1, depth + 1):
            conv = nn.Sequential(
                nn.Conv2d(in_channels, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=n_groups),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
                nn.Conv2d(dim, out_channels, kernel_size=1),
                nn.Dropout(drop),
            )
            setattr(self, 'conv%d' % i, conv)
            in_channels = out_channels

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, ratio, n_groups, depth, drop=0.):
        super(Down, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Basic_layer(out_channels, out_channels, ratio, n_groups, depth, drop)
        )


class Stem(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        times = int(math.log(patch_size[0], 2))
        self.proj = nn.ModuleList(self.layer(
            in_chans=in_chans if i == 0 else embed_dim // 2 ** (times - i),
            dim=embed_dim // 2 ** (times - i - 1)
        ) for i in range(times))

    def layer(self, in_chans, dim):
        return nn.Sequential(
            nn.Conv2d(in_chans, dim // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        for layer in self.proj:
            x = layer(x)
        return x


class PPM(nn.Module):
    def __init__(self, ppm_in_chans, out_chans, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.pool_projs = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveMaxPool2d(pool_size),
                nn.Conv2d(ppm_in_chans, out_chans, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(True)
            )for pool_size in pool_sizes)

        self.bottom = nn.Sequential(
            nn.Conv2d(ppm_in_chans + len(pool_sizes) * out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(True)
        )

    def forward(self, x):
        xs = [x]
        for pool_proj in self.pool_projs:
            pool_x = F.interpolate(pool_proj(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            xs.append(pool_x)

        x = torch.cat(xs, dim=1)
        x = self.bottom(x)

        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU(True)
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, C, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.transpose(1, 2).contiguous().view(-1, C, H, W)


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, bias=qkv_bias, padding=1, groups=num_heads),
            nn.BatchNorm2d(dim)
        )
        self.proj = SeparableConvBN(dim, dim, kernel_size=3)
        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]
        out = attn + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., drop_path=0., window_size=8):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        _, C, H, W = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x.flatten(2).transpose(1, 2)), C, H, W))

        return x


class Fuse(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128):
        super(Fuse, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weight = nn.Parameter(torch.ones(2))
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, en):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        weight = F.softmax(self.weight)
        x = self.pre_conv(en) * weight[0] + x * weight[1]
        x = self.post_conv(x)
        return x


class Decoder_Head(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64, chan_ratio=16):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, 1, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Sigmoid()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(decode_channels, decode_channels // chan_ratio, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(decode_channels // chan_ratio, decode_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.GELU()

    def forward(self, x, en):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.pre_conv(en) + x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x
