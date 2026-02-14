import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

class ContextBlock(nn.Module):
    """ Global Context Block """

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x.view(batch, channel, height * width).unsqueeze(1)  # [N, 1, C, H * W]
            context_mask = self.conv_mask(x).view(batch, 1, height * width)
            context_mask = self.softmax(context_mask).unsqueeze(-1)  # [N, 1, H * W, 1]
            context = torch.matmul(input_x, context_mask).view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class GuidedAttention(nn.Module):
    """ Reconstruction Guided Attention with Ground-truth Mask Integration. """

    def __init__(self, depth=1792, drop_rate=0.2):
        super(GuidedAttention, self).__init__()
        self.depth = depth

        self.gated = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

        self.h = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, pred_x, embedding, mask=None):
        residual_full = torch.abs(x - pred_x)
        target_size = embedding.shape[-2:]
        residual_x = F.interpolate(residual_full, size=target_size, mode='bilinear', align_corners=True)

        if mask is not None:
            mask_resized = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=True)
            mask_feat = self.mask_conv(mask_resized)
            residual_x = residual_x + mask_feat

        res_map = self.gated(residual_x)
        out = res_map * self.h(embedding) + self.dropout(embedding)
        return out

class PPM(nn.Module):
    def __init__(self, in_dim, out_dim, bins):
        super(PPM, self).__init__()
        self.features = nn.ModuleList()
        self.conv_reduce = nn.Conv2d(1240, 728, kernel_size=1)

        for bin_size in bins:
            if bin_size == 1:
                self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True)
                ))

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            temp = f(x)
            temp = F.interpolate(temp, size=x_size[2:], mode="bilinear", align_corners=True)
            out.append(temp)
        output = torch.cat(out, 1)
        output = self.conv_reduce(output)
        return output


class PPM_Normal(nn.Module):
    """ Normal Pyramidal Pooling Module without reduction block. """

    def __init__(self, in_dim, out_dim, bins):
        super(PPM_Normal, self).__init__()
        self.features = nn.ModuleList()
        for bin_size in bins:
            if bin_size == 1:
                self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True)
                ))

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            temp = f(x)
            temp = F.interpolate(temp, x_size[2:], mode="bilinear", align_corners=True)
            out.append(temp)
        output = torch.cat(out, 1)
        return output


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels

            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(outc))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class ReXception_PPM(nn.Module):
    """ Reconstruction Branch using Xception and PPM. """

    def __init__(self, num_classes=2, drop_rate=0.2, training=True):
        super(ReXception_PPM, self).__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(drop_rate)
        self.training = training

        self.ppm = PPM(in_dim=728, out_dim=128, bins=[1, 2, 4, 8])

        # --- Decoder ---
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        # --- Entry flow ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=True)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)

        self.block4 = Block(728, 728, 3, 1)
        self.block5 = Block(728, 728, 3, 1)
        self.block6 = Block(728, 728, 3, 1)
        self.block7 = Block(728, 728, 3, 1)

        self.block8 = Block(728, 728, 3, 1)
        self.block9 = Block(728, 728, 3, 1)
        self.block10 = Block(728, 728, 3, 1)
        self.block11 = Block(728, 728, 3, 1)

        self.block12 = Block(728, 1024, 2, 2, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.act4 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(2048, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = torch.rand([tensor.shape[0], 1, 1, 1])
        rand = torch.where(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        noise_t = torch.clip(noise_t, -1., 1.)
        return noise_t

    def forward(self, x):
        noise_x = self.add_white_noise(x) if self.training else x

        out = self.conv1(noise_x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        embedding = self.block4(out)

        out = self.dropout(embedding)
        out = self.ppm(out)

        out = self.decoder1(out)
        out_d2 = self.decoder2(out)

        out = self.decoder3(out_d2)
        out_d4 = self.decoder4(out)

        out = self.decoder5(out_d4)
        pred = self.decoder6(out)

        re_img = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return re_img

class MDL_Model(nn.Module):
    """
    Main Multi-Branch Forgery Detection Model.
    Integrates EfficientNet, Reconstruction Branch, and Guided Attention.
    """

    def __init__(self, num_classes=2, backbone_name='efficientnet-b4', pretrained=True):
        super(MDL_Model, self).__init__()

        self.eff = EfficientNet.from_pretrained(backbone_name) if pretrained else EfficientNet.from_name(backbone_name)

        self.re_xception = ReXception_PPM(num_classes=2)
        self.attention = GuidedAttention(depth=1792, drop_rate=0.2)

        self.ppm = PPM_Normal(in_dim=1792, out_dim=448, bins=[1, 2, 4, 8])
        self.gcatt = ContextBlock(inplanes=3584, ratio=1. / 16., pooling_type='att')

        self.conv1x1 = nn.Conv2d(in_channels=3584, out_channels=2, kernel_size=1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1792, num_classes)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input image [B, 3, H, W]
            mask: Optional ground truth mask for attention guidance [B, 1, H, W]
        Returns:
            logits: Classification predictions [B, num_classes]
            re_img: Reconstructed image [B, 3, H, W]
            seg_output: Segmentation/Localization map [B, 2, H, W]
        """
        feat = self.eff.extract_features(x)

        ppm_feat = self.ppm(feat)  # [B, 3584, h, w]
        gc_feat = self.gcatt(ppm_feat)  # [B, 3584, h, w]

        seg_map = self.conv1x1(gc_feat)  # [B, 2, h, w]
        seg_output = F.interpolate(seg_map, size=x.shape[2:], mode='bilinear', align_corners=False)

        re_img = self.re_xception(x)  # [B, 3, H, W]

        att_feat = self.attention(x, re_img, feat, mask=mask)

        out = self.avg_pooling(att_feat).flatten(1)
        out = self.dropout(out)
        logits = self.fc(out)

        return logits, re_img, seg_output

if __name__ == "__main__":
    model = MDL_Model(num_classes=2, pretrained=False)

    dummy_x = torch.randn(2, 3, 256, 256)
    dummy_mask = torch.randn(2, 1, 256, 256)

    logits, re_img, seg_output = model(dummy_x, mask=dummy_mask)

    print("Forward Pass Successful!")
    print(f"Logits shape: {logits.shape}")
    print(f"Reconstructed Image: {re_img.shape}")
    print(f"Segmentation Output: {seg_output.shape}")