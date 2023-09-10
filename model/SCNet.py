import torch
import torch.nn as nn
import torch.nn.functional as F


class SCNet(nn.Module):
    def __init__(self, n_classes, in_chans=3, dim=64):
        super(SCNet, self).__init__()
        self.n_classes = n_classes
        self.aux = False
        c1, c2, c3, c4, c5 = [dim * 2 ** i for i in range(5)]

        self.inc = DoubleConv(in_chans, c3)
        self.down1 = Down()
        self.down2 = Down()
        self.down3 = Down()
        self.down4 = Down()
        en_dim = c3

        self.attn_encoder = Atten_encoder(en_dim)

        self.up4 = Up(en_dim, c4, en_dim)
        self.up3 = Up(c4, c3, en_dim)
        self.up2 = Up(c3, c2, en_dim)
        self.up1 = Up(c2, c1, en_dim)

        self.out1 = OutConv(c1, n_classes)
        self.out2 = OutConv(c2, n_classes)
        self.out3 = OutConv(c3, n_classes)
        self.out4 = OutConv(c4, n_classes)


    def forward(self, x):
        x = x if isinstance(x, tuple or list) else tuple([x])
        x = torch.cat(x, dim=1)
        h, w = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.attn_encoder(x2)

        x3 = self.down2(x2)
        x3 = self.attn_encoder(x3)

        x4 = self.down3(x3)
        x4 = self.attn_encoder(x4)

        x5 = self.down4(x4)
        x5 = self.attn_encoder(x5)

        x = self.up4(x5, x4)
        x4 = self.out4(x)
        x4 = F.interpolate(x4, size=(h, w), mode='nearest')
        x = self.up3(x, x3)
        x3 = self.out3(x)
        x3 = F.interpolate(x3, size=(h, w), mode='nearest')
        x = self.up2(x, x2)
        x2 = self.out2(x)
        x2 = F.interpolate(x2, size=(h, w), mode='nearest')
        x = self.up1(x, x1)
        x1 = self.out1(x)
        x = (x1 + x2 + x3 + x4) / 4

        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class CBAM_channel_attention(nn.Module):
    def __init__(self, in_chans, ratio=8, act_layer=nn.ReLU):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False)
        self.act_layer = act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        self.conv2 = nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.act_layer(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.act_layer(self.conv1(self.max_pool(x))))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight


class Atten_encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            CBAM_channel_attention(out_channels)
        )

    def forward(self, x):
        return self.attn(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self):
        super().__init__()
        self.maxpool_conv = nn.MaxPool2d(2)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, t):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(t + out_channels, out_channels)

    def forward(self, x, x1):
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

