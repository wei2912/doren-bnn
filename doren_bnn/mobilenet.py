from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    AdaptiveAvgPool2d,
    Linear,
)

from enum import Enum

from .xnorpp import Conv2d_XnorPP
from .xnor_react import Conv2d_Xnor_ReAct, RPReLU


class NetType(Enum):
    REAL = "Real"
    XNORPP = "XnorPP"
    XNOR_REACT = "Xnor-ReAct"


class MobileNet_ConvBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        in_size: int,
        nettype: NetType,
    ):
        super(MobileNet_ConvBlock, self).__init__()

        block_params = {"stride": stride, "padding": 1, "bias": False}

        self.block = {
            NetType.REAL: Sequential(
                Conv2d(in_channels, out_channels, 3, **block_params),
                BatchNorm2d(out_channels),
                ReLU(inplace=True),
            ),
            NetType.XNORPP: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_XnorPP(in_channels, out_channels, 3, in_size, **block_params),
                ReLU(inplace=True),
            ),
            NetType.XNOR_REACT: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_Xnor_ReAct(in_channels, out_channels, 3, **block_params),
                RPReLU(),
            ),
        }[nettype]

    def forward(self, input: Tensor) -> Tensor:
        return self.block(input)


class MobileNet_ConvDsBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        in_size: int,
        nettype: NetType,
    ):
        super(MobileNet_ConvDsBlock, self).__init__()

        block_dw_params = {
            "stride": stride,
            "padding": 1,
            "groups": in_channels,
            "bias": False,
        }
        self.block_dw = {
            NetType.REAL: Sequential(
                Conv2d(in_channels, in_channels, 3, **block_dw_params),
                BatchNorm2d(in_channels),
                ReLU(inplace=True),
            ),
            NetType.XNORPP: Sequential(
                BatchNorm2d(in_channels),
                Conv2d(in_channels, in_channels, 3, **block_dw_params),
                ReLU(inplace=True),
            ),
            NetType.XNOR_REACT: Sequential(
                BatchNorm2d(in_channels),
                Conv2d(in_channels, in_channels, 3, **block_dw_params),
                RPReLU(),
            ),
        }[nettype]

        block_pw_params = {"stride": 1, "padding": 0, "bias": False}
        self.block_pw = {
            NetType.REAL: Sequential(
                Conv2d(in_channels, out_channels, 1, **block_pw_params),
                BatchNorm2d(out_channels),
                ReLU(inplace=True),
            ),
            NetType.XNORPP: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_XnorPP(
                    in_channels, out_channels, 1, in_size // stride, **block_pw_params
                ),
                ReLU(inplace=True),
            ),
            NetType.XNOR_REACT: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_Xnor_ReAct(in_channels, out_channels, 1, **block_pw_params),
                RPReLU(),
            ),
        }[nettype]

        self.block = Sequential(self.block_dw, self.block_pw)

    def forward(self, input: Tensor) -> Tensor:
        return self.block(input)


class MobileNet(Module):
    def __init__(
        self, in_channels: int, in_size: int, num_classes: int = 1000, **kwargs
    ):
        super(MobileNet, self).__init__()

        kwargs_real = {**kwargs}
        kwargs_real["nettype"] = NetType.REAL

        self.model = Sequential(
            MobileNet_ConvBlock(in_channels, 32, 2, in_size, **kwargs_real),
            MobileNet_ConvDsBlock(32, 64, 1, in_size // 2, **kwargs),
            MobileNet_ConvDsBlock(64, 128, 2, in_size // 2, **kwargs),
            MobileNet_ConvDsBlock(128, 128, 1, in_size // 4, **kwargs),
            MobileNet_ConvDsBlock(128, 256, 2, in_size // 4, **kwargs),
            MobileNet_ConvDsBlock(256, 256, 1, in_size // 8, **kwargs),
            MobileNet_ConvDsBlock(256, 512, 2, in_size // 8, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, in_size // 16, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, in_size // 16, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, in_size // 16, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, in_size // 16, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, in_size // 16, **kwargs),
            MobileNet_ConvDsBlock(512, 1024, 2, in_size // 16, **kwargs),
            MobileNet_ConvDsBlock(1024, 1024, 1, in_size // 32, **kwargs_real),
            AdaptiveAvgPool2d(1),
        )
        self.fc = Linear(1024, num_classes)

    def forward(self, input: Tensor) -> Tensor:
        input = self.model(input).view(-1, 1024)
        return self.fc(input)
