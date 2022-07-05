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
from .xnorpp_sca import Conv2d_XnorPP_SCA


class NetType(Enum):
    REAL = "Real"
    XNORPP = "XnorPP"
    XNOR_REACT = "Xnor-ReAct"
    XNORPP_SCA = "XnorPP-SCA"


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
    ):
        super(MobileNet_ConvBlock, self).__init__()
        self.nettype = nettype

        block_params = {"stride": stride, "padding": 1, "bias": False}

        self.block = {
            NetType.REAL: Sequential(
                Conv2d(in_channels, out_channels, 3, **block_params),
                BatchNorm2d(out_channels),
                ReLU(inplace=True),
            ),
            NetType.XNORPP: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_XnorPP(in_channels, out_channels, 3, **block_params),
                ReLU(inplace=True),
            ),
            NetType.XNOR_REACT: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_Xnor_ReAct(in_channels, out_channels, 3, **block_params),
                RPReLU(),
            ),
            NetType.XNORPP_SCA: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_XnorPP_SCA(in_channels, out_channels, 3, **block_params),
                ReLU(inplace=True),
            ),
        }[nettype]

    def forward(self, input: Tensor) -> Tensor:
        return self.block(input)

    def wdr(self, alpha: float) -> Tensor:
        if not self.nettype == NetType.XNORPP_SCA:
            return 0

        return self.block[1].wdr(alpha)


class MobileNet_ConvDsBlock(MobileNet_Block):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        nettype: NetType,
    ):
        super(MobileNet_ConvDsBlock, self).__init__()
        self.nettype = nettype

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
                Conv2d_XnorPP(in_channels, in_channels, 3, **block_dw_params),
                ReLU(inplace=True),
            ),
            NetType.XNOR_REACT: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_Xnor_ReAct(in_channels, in_channels, 3, **block_dw_params),
                RPReLU(),
            ),
            NetType.XNORPP_SCA: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_XnorPP_SCA(in_channels, in_channels, 3, **block_dw_params),
                ReLU(inplace=True),
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
                Conv2d_XnorPP(in_channels, out_channels, 1, **block_pw_params),
                ReLU(inplace=True),
            ),
            NetType.XNOR_REACT: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_Xnor_ReAct(in_channels, out_channels, 1, **block_dw_params),
                RPReLU(),
            ),
            NetType.XNORPP_SCA: Sequential(
                BatchNorm2d(in_channels),
                Conv2d_XnorPP_SCA(in_channels, out_channels, 1, **block_pw_params),
                ReLU(inplace=True),
            ),
        }[nettype]

        self.block = Sequential(self.block_dw, self.block_pw)

    def forward(self, input: Tensor) -> Tensor:
        return self.block(input)

    def wdr(self, alpha: float) -> Tensor:
        if not self.nettype == NetType.XNORPP_SCA:
            return 0

        return self.block_dw[1].wdr(alpha) + self.block_pw[1].wdr(alpha)


class MobileNet(Module):
    def __init__(self, in_channels: int, num_classes: int = 1000, **kwargs):
        super(MobileNet, self).__init__()

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
        input = self.model(input).view(-1, 1024)
        return self.fc(input)

    def wdr(self, alpha: float) -> Tensor:
        wdrs = [
            layer.wdr(alpha) if isinstance(layer, MobileNet_Block) else 0.0
            for layer in self.model
        ]
        # print(["{:.3f}".format(float(wdr)) for wdr in wdrs])
        return sum(wdrs)
