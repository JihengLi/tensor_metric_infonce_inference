import torch
from torch import nn


class SEBlock(nn.Module):
    """Squeeze and Excitation block"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1, 1)
        return x * y


class ResidualSEBlock(nn.Module):
    """Block for ResNet3D with SE attention"""

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        reduction=16,
        drop_rate=0.0,
    ):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock(out_channels, reduction=reduction)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else None
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """3D ResNet encoder with SE attention"""

    def __init__(
        self,
        block=ResidualSEBlock,
        layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512),
        num_channels=6,
        proj_hidden_dim=512,
        emb_dim=128,
        reduction=16,
        drop_rate=0.0,
    ):
        super(Encoder, self).__init__()
        self.in_channels = channels[0]
        # 7x7x7 conv -> BN -> ReLU -> 3x3x3 max pool
        self.conv1 = nn.Conv3d(
            num_channels,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # Residual layers
        # layer1: output channels = channels[0], stride=1
        self.layer1 = self._make_layer(
            block,
            channels[0],
            layers[0],
            stride=1,
            reduction=reduction,
            drop_rate=drop_rate,
        )
        # layer2: output channels = channels[1], stride=2 (downsample)
        self.layer2 = self._make_layer(
            block,
            channels[1],
            layers[1],
            stride=2,
            reduction=reduction,
            drop_rate=drop_rate,
        )
        # layer3: output channels = channels[2], stride=2
        self.layer3 = self._make_layer(
            block,
            channels[2],
            layers[2],
            stride=2,
            reduction=reduction,
            drop_rate=drop_rate,
        )
        # layer4: output channels = channels[3], stride=2
        self.layer4 = self._make_layer(
            block,
            channels[3],
            layers[3],
            stride=2,
            reduction=reduction,
            drop_rate=drop_rate,
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.emb_dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else None
        self.fc = nn.Sequential(
            nn.Linear(channels[3] * block.expansion, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def _make_layer(
        self, block, out_channels, blocks, stride=1, reduction=16, drop_rate=0.0
    ):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                reduction=reduction,
                drop_rate=drop_rate,
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    downsample=None,
                    reduction=reduction,
                    drop_rate=drop_rate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch, 4, D, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Global average pool to (batch, channels, 1, 1, 1)
        x = self.global_pool(x)
        # flatten to (batch, channels)
        x = torch.flatten(x, 1)
        if self.emb_dropout is not None:
            x = self.emb_dropout(x)
        x = self.fc(x)
        return x
