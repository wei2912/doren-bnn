import torch

from torch import Tensor
from torch.nn import Conv2d, Module, Parameter
import torch.nn.functional as F

from .xnorpp import Sign


class RSign(Module):
    r"""
    Implement module for the RSign function from ReActNet.

    For an input $r$ and learnable parameter $\alpha$,
    Forward pass: $\text{sign}(r)$, where $\text{sign}(r) = 1$ if $r > \alpha$ and
    $-1$ otherwise.
    """

    def __init__(self, in_channels: int):
        super(RSign, self).__init__()

        self.alpha = Parameter(torch.zeros(in_channels).reshape(-1, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return Sign.apply(input - self.alpha)


class RPReLU(Module):
    r"""
    Implement module for the RPReLU function from ReActNet.

    For an input $r$ and learnable parameters $\beta$, $\gamma$ and $\zeta$,
    Forward pass: $r - \gamma + \zeta$, if $r > \gamma$, and $\beta(r - \gamma) + \zeta$
    otherwise.
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super(RPReLU, self).__init__()

        self.beta = Parameter(torch.empty(num_parameters).fill_(init).reshape(-1, 1, 1))
        self.gamma = Parameter(torch.zeros(num_parameters).reshape(-1, 1, 1))
        self.zeta = Parameter(torch.zeros(num_parameters).reshape(-1, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input - self.gamma, self.beta) + self.zeta


class Conv2d_Xnor_ReAct(Module):
    """
    Implement convolutional layer with binary weights and activations, following
    ReActNet.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super(Conv2d_Xnor_ReAct, self).__init__()

        if kwargs["bias"]:
            raise NotImplementedError("bias is not supported on Conv2d_ReAct")
        del kwargs["bias"]

        self.rsign = RSign(in_channels)

        self.conv2d_params = kwargs
        conv2d = Conv2d(
            in_channels, out_channels, kernel_size, bias=False, **self.conv2d_params
        )

        self.weight = Parameter(conv2d.weight.detach())

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(
            self.rsign(input), Sign.apply(self.weight), **self.conv2d_params
        )
