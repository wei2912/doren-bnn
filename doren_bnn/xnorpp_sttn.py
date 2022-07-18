from torch import Tensor
from torch.nn import Conv2d, Module, Parameter
import torch.nn.functional as F

from .xnorpp import Sign


class Conv2d_XnorPP_STTN(Module):
    """
    Implement convolutional layer with binary weights and ternary activations, following
    Case 1 of XNOR-Net++.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        learnable: bool = False,
        **kwargs
    ):
        super(Conv2d_XnorPP_STTN, self).__init__()

        if kwargs["bias"]:
            raise NotImplementedError("bias is not supported on Conv2d_XnorPP")
        del kwargs["bias"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv2d_params = kwargs
        conv2d1 = Conv2d(
            in_channels, out_channels, kernel_size, bias=False, **self.conv2d_params
        )
        conv2d2 = Conv2d(
            in_channels, out_channels, kernel_size, bias=False, **self.conv2d_params
        )

        self.weight1 = Parameter(conv2d1.weight)
        self.weight2 = Parameter(conv2d2.weight)

        self.learnable = learnable
        if self.learnable:
            self.alpha = Parameter(self._calc_channel_scaling())

    def forward(self, input: Tensor) -> Tensor:
        output = F.conv2d(
            Sign.apply(input), self._get_weight_ter(), **self.conv2d_params
        )
        return output.mul(
            self.alpha
            if self.learnable
            else self._calc_channel_scaling().to(input.device)
        )

    def _calc_channel_scaling(self) -> Tensor:
        return (
            self.weight1.abs().mean((1, 2, 3)) + self.weight2.abs().mean((1, 2, 3))
        ).reshape(-1, 1, 1)

    def _get_weight_ter(self) -> Tensor:
        return (Sign.apply(self.weight1) + Sign.apply(self.weight2)) / 2.0
