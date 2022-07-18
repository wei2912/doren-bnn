from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    AdaptiveAvgPool2d,
    Linear,
    Upsample,
)

from enum import Enum

from .xnorpp import Conv2d_XnorPP
from .xnorpp_sca import Conv2d_XnorPP_SCA
from .xnorpp_sttn import Conv2d_XnorPP_STTN


class NetType(Enum):
    REAL = "Real"
    XNORPP = "XnorPP"
    XNORPP_SCA = "XnorPP-SCA"
    XNORPP_STTN = "XnorPP-STTN"


class MobileNet_Block(Module):
    def __init__(self):
        super(MobileNet_Block, self).__init__()


class MobileNet_ConvBlock(MobileNet_Block):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        nettype: NetType,
        **kwargs,
    ):
        super(MobileNet_ConvBlock, self).__init__()

        block_params = {"stride": stride, "padding": 1, "bias": False, **kwargs}

        match nettype:
            case NetType.REAL:
                self.block = Sequential(
                    Conv2d(in_channels, out_channels, 3, **block_params),
                    BatchNorm2d(out_channels),
                    ReLU(inplace=True),
                )
            case NetType.XNORPP:
                self.block = Sequential(
                    BatchNorm2d(in_channels),
                    Conv2d_XnorPP(in_channels, out_channels, 3, **block_params),
                    ReLU(inplace=True),
                )
            case NetType.XNORPP_SCA:
                self.block = Sequential(
                    BatchNorm2d(in_channels),
                    Conv2d_XnorPP_SCA(in_channels, out_channels, 3, **block_params),
                    ReLU(inplace=True),
                )
            case NetType.XNORPP_STTN:
                self.block = Sequential(
                    BatchNorm2d(in_channels),
                    Conv2d_XnorPP_STTN(in_channels, out_channels, 3, **block_params),
                    ReLU(inplace=True),
                )
            case _:
                raise NotImplementedError(f"nettype {nettype} not supported")

    def forward(self, input: Tensor) -> Tensor:
        return self.block(input)


class MobileNet_ConvDsBlock(MobileNet_Block):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        nettype: NetType,
        **kwargs,
    ):
        super(MobileNet_ConvDsBlock, self).__init__()

        block_dw_params = {
            "stride": stride,
            "padding": 1,
            "groups": in_channels,
            "bias": False,
            **kwargs,
        }
        match nettype:
            case NetType.REAL:
                self.block_dw = Sequential(
                    Conv2d(in_channels, in_channels, 3, **block_dw_params),
                    BatchNorm2d(in_channels),
                    ReLU(inplace=True),
                )
            case NetType.XNORPP:
                self.block_dw = Sequential(
                    BatchNorm2d(in_channels),
                    Conv2d_XnorPP(in_channels, in_channels, 3, **block_dw_params),
                    ReLU(inplace=True),
                )
            case NetType.XNORPP_SCA:
                self.block_dw = Sequential(
                    BatchNorm2d(in_channels),
                    Conv2d_XnorPP_SCA(in_channels, in_channels, 3, **block_dw_params),
                    ReLU(inplace=True),
                )
            case NetType.XNORPP_STTN:
                self.block_dw = Sequential(
                    BatchNorm2d(in_channels),
                    Conv2d_XnorPP_STTN(in_channels, in_channels, 3, **block_dw_params),
                    ReLU(inplace=True),
                )
            case _:
                raise NotImplementedError(f"nettype {nettype} not supported")

        block_pw_params = {"stride": 1, "padding": 0, "bias": False, **kwargs}
        match nettype:
            case NetType.REAL:
                self.block_pw = Sequential(
                    Conv2d(in_channels, out_channels, 1, **block_pw_params),
                    BatchNorm2d(out_channels),
                    ReLU(inplace=True),
                )
            case NetType.XNORPP:
                self.block_pw = Sequential(
                    BatchNorm2d(in_channels),
                    Conv2d_XnorPP(in_channels, out_channels, 1, **block_pw_params),
                    ReLU(inplace=True),
                )
            case NetType.XNORPP_SCA:
                self.block_pw = Sequential(
                    BatchNorm2d(in_channels),
                    Conv2d_XnorPP_SCA(in_channels, out_channels, 1, **block_pw_params),
                    ReLU(inplace=True),
                )
            case NetType.XNORPP_STTN:
                self.block_pw = Sequential(
                    BatchNorm2d(in_channels),
                    Conv2d_XnorPP_STTN(in_channels, out_channels, 1, **block_pw_params),
                    ReLU(inplace=True),
                )
            case _:
                raise NotImplementedError(f"nettype {nettype} not supported")

        self.block = Sequential(self.block_dw, self.block_pw)

    def forward(self, input: Tensor) -> Tensor:
        return self.block(input)


class MobileNet(Module):
    def __init__(self, in_channels: int, num_classes: int = 1000, **kwargs):
        super(MobileNet, self).__init__()

        self.upsample = Upsample((224, 224))
        self.model = Sequential(
            MobileNet_ConvBlock(in_channels, 32, 2, **kwargs),
            MobileNet_ConvDsBlock(32, 64, 1, **kwargs),
            MobileNet_ConvDsBlock(64, 128, 2, **kwargs),
            MobileNet_ConvDsBlock(128, 128, 1, **kwargs),
            MobileNet_ConvDsBlock(128, 256, 2, **kwargs),
            MobileNet_ConvDsBlock(256, 256, 1, **kwargs),
            MobileNet_ConvDsBlock(256, 512, 2, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, **kwargs),
            MobileNet_ConvDsBlock(512, 512, 1, **kwargs),
            MobileNet_ConvDsBlock(512, 1024, 2, **kwargs),
            MobileNet_ConvDsBlock(1024, 1024, 1, **kwargs),
            AdaptiveAvgPool2d(1),
        )
        self.fc = Linear(1024, num_classes)

    def forward(self, input: Tensor) -> Tensor:
        up_input = self.upsample(input)
        output = self.model(up_input).view(-1, 1024)
        return self.fc(output)
