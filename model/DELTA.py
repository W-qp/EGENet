import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_chans, depth, out_chans, n_classes):
        super().__init__()
        self.depth = depth
        stage = [i for i in range(depth)]

        self.conv = nn.Sequential(
                nn.Conv2d(in_chans * 2 ** 3, out_chans, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(True),
            )
        self.conv_ = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_chans * 2 ** stage[::-1][i + 1], out_chans, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(True),
            )for i in range(depth - 1))

        self.fpn_conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(True),
            )for _ in range(depth - 1))

        self.out = nn.Conv2d(out_chans * depth, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = x[::-1]
        fpn_x = self.conv(x[0])
        x[0] = fpn_x
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


class DELTA(nn.Module):
    def __init__(self,
                 n_classes,
                 in_chans=3,
                 decode_channels=256,
                 aux=False):
        super().__init__()
        self.aux = aux
        self.n_classes = n_classes
        self.backbone = timm.create_model('resnet50', in_chans=in_chans, features_only=True, out_indices=(1, 2, 3, 4))
        encoder_channels = self.backbone.feature_info.channels()[0]
        self.decoder = FPN(encoder_channels, 4, decode_channels, n_classes)

    def forward(self, x):
        x = x if isinstance(x, tuple or list) else tuple([x])
        x = torch.cat(x, dim=1)
        h, w = x.shape[2:]

        x = self.backbone(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x
