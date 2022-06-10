from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Linear,
    PReLU,
)

from enum import Enum

from doren_bnn.xnor_net import Conv2d_XNorNetPP


class NetType(Enum):
    REAL = "Real"
    BI_REAL = "Bi-Real"
    BI_REAL_GROUPED = "Bi-Real_Grouped"


class InitialBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        nettype: NetType = NetType.REAL,
    ) -> None:
        super(InitialBlock, self).__init__()

        conv_params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "groups": groups,
            "bias": False,
        }

        self.nettype = nettype
        self.block = {
            NetType.REAL: Sequential(
                Conv2d(in_channels, out_channels, **conv_params),
                BatchNorm2d(out_channels),
            ),
            NetType.BI_REAL: Sequential(
                Conv2d_XNorNetPP(in_channels, out_channels, **conv_params),
                BatchNorm2d(out_channels),
            ),
            NetType.BI_REAL_GROUPED: Sequential(
                Conv2d_XNorNetPP(in_channels, out_channels, **conv_params),
                BatchNorm2d(out_channels),
            ),
        }[nettype]
        self.prelu = PReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.nettype in [NetType.REAL, NetType.BI_REAL]:
            return self.prelu(self.block(x))


class BasicBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        nettype: NetType = NetType.REAL,
    ) -> None:
        super(BasicBlock, self).__init__()

        conv_params = {
            "kernel_size": kernel_size,
            "padding": padding,
            "groups": groups,
            "bias": False,
        }

        self.nettype = nettype
        self.block1 = {
            NetType.REAL: Sequential(
                Conv2d(in_channels, out_channels, stride=stride, **conv_params),
                BatchNorm2d(out_channels),
            ),
            NetType.BI_REAL: Sequential(
                Conv2d_XNorNetPP(
                    in_channels, out_channels, stride=stride, **conv_params
                ),
                BatchNorm2d(out_channels),
            ),
            NetType.BI_REAL_GROUPED: Sequential(
                Conv2d_XNorNetPP(
                    in_channels, out_channels, stride=stride, **conv_params
                ),
                BatchNorm2d(out_channels),
            ),
        }[nettype]
        self.prelu1 = PReLU()
        self.block2 = {
            NetType.REAL: Sequential(
                Conv2d(out_channels, out_channels, stride=1, **conv_params),
                BatchNorm2d(out_channels),
            ),
            NetType.BI_REAL: Sequential(
                Conv2d_XNorNetPP(out_channels, out_channels, stride=1, **conv_params),
                BatchNorm2d(out_channels),
            ),
            NetType.BI_REAL_GROUPED: Sequential(
                Conv2d_XNorNetPP(out_channels, out_channels, stride=1, **conv_params),
                BatchNorm2d(out_channels),
            ),
        }[nettype]
        self.prelu2 = PReLU()

        if stride == 1:
            self.downsample = None
        else:
            downsample_params = {"kernel_size": 1, "stride": stride, "bias": False}
            self.downsample = Sequential(
                Conv2d(in_channels, out_channels, **downsample_params),
                BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.nettype == NetType.REAL:
            identity = x if self.downsample is None else self.downsample(x)
            out2 = self.block2(self.prelu1(self.block1(x)))
            return self.prelu2(out2 + identity)
        elif self.nettype == NetType.BI_REAL or self.nettype == NetType.BI_REAL_GROUPED:
            identity1 = x if self.downsample is None else self.downsample(x)
            out1 = self.prelu1(self.block1(x) + identity1)
            return self.prelu2(self.block2(out1) + out1)


class ResNet18(Module):
    def __init__(self, num_classes: int = 1000, **kwargs):
        super(ResNet18, self).__init__()

        kwargs_real = kwargs.copy()
        kwargs_real["nettype"] = NetType.REAL

        nettype = kwargs["nettype"]

        groups = (
            [1, 1, 1, 1] if nettype != NetType.BI_REAL_GROUPED else [16, 32, 32, 64]
        )

        self.block1 = InitialBlock(
            3, 64, kernel_size=7, stride=2, padding=3, **kwargs_real
        )
        self.block2 = Sequential(
            BasicBlock(64, 64, kernel_size=3, padding=1, groups=groups[0], **kwargs),
            BasicBlock(64, 64, kernel_size=3, padding=1, groups=groups[0], **kwargs),
        )
        self.block3 = Sequential(
            BasicBlock(
                64, 128, kernel_size=3, stride=2, padding=1, groups=groups[1], **kwargs
            ),
            BasicBlock(128, 128, kernel_size=3, padding=1, groups=groups[1], **kwargs),
        )
        self.block4 = Sequential(
            BasicBlock(
                128, 256, kernel_size=3, stride=2, padding=1, groups=groups[2], **kwargs
            ),
            BasicBlock(256, 256, kernel_size=3, padding=1, groups=groups[2], **kwargs),
        )
        self.block5 = Sequential(
            BasicBlock(
                256, 512, kernel_size=3, stride=2, padding=1, groups=groups[3], **kwargs
            ),
            BasicBlock(512, 512, kernel_size=3, padding=1, groups=groups[3], **kwargs),
        )

        self.model = Sequential(
            self.block1,
            MaxPool2d(3, stride=2, padding=1),
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            AdaptiveAvgPool2d(1),
        )
        self.fc = Linear(512, num_classes)

    def forward(self, input: Tensor) -> Tensor:
        input = self.model(input).view(-1, 512)
        return self.fc(input)
