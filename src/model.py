import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x_down = self.pool(x)
        return x, x_down


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False
        )
        self.conv1 = ConvBNReLU(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TinyUNetMultiTask(nn.Module):
    """
    Two-head network:
      1) Segmentation: seg_logits (B,1,H,W)
      2) Classification: cls_logits (B,num_classes)

    Detection is derived from the predicted segmentation mask externally.
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 10, base: int = 32):
        super().__init__()

        # Encoder
        self.d1 = DownBlock(in_channels, base)
        self.d2 = DownBlock(base, base * 2)
        self.d3 = DownBlock(base * 2, base * 4)
        self.d4 = DownBlock(base * 4, base * 8)

        # Bottleneck
        self.b1 = ConvBNReLU(base * 8, base * 16)
        self.b2 = ConvBNReLU(base * 16, base * 16)

        # Segmentation decoder
        self.u4 = UpBlock(base * 16, base * 8, base * 8)
        self.u3 = UpBlock(base * 8, base * 4, base * 4)
        self.u2 = UpBlock(base * 4, base * 2, base * 2)
        self.u1 = UpBlock(base * 2, base, base)
        self.seg_head = nn.Conv2d(base, 1, kernel_size=1)

        # Shared global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)


        self.cls_fc = nn.Sequential(
            nn.Linear(base * 16 + base + base, base * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(base * 8, base * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(base * 4, num_classes),
        )

    def _attention_pool(self, feat_map: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:

        weighted = feat_map * att_map
        numerator = weighted.sum(dim=(2, 3))          # (B,C)
        denominator = att_map.sum(dim=(2, 3)) + 1e-6  # (B,1)
        return numerator / denominator

    def forward(self, x: torch.Tensor):
        # Encoder
        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        s4, x = self.d4(x)

        # Bottleneck
        x = self.b1(x)
        x = self.b2(x)
        bottleneck_feat_map = x

        # Segmentation decoder
        x = self.u4(bottleneck_feat_map, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        decoder_feat_map = x


        seg_logits = self.seg_head(decoder_feat_map)


        seg_att = torch.sigmoid(seg_logits).detach()  # (B,1,H,W)


        bottleneck_feat = self.global_pool(bottleneck_feat_map).flatten(1)  # (B, base*16)


        decoder_feat_att = self._attention_pool(decoder_feat_map, seg_att)  # (B, base)


        decoder_feat_global = self.global_pool(decoder_feat_map).flatten(1) # (B, base)


        fused_feat = torch.cat(
            [bottleneck_feat, decoder_feat_att, decoder_feat_global], dim=1
        )


        cls_logits = self.cls_fc(fused_feat)

        return seg_logits, cls_logits
