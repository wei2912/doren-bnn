from torch import Tensor
from torch.nn import Conv2d, Module, Parameter
import torch.nn.functional as F

from .xnorpp import Sign


class Conv2d_XnorPP_STTN(Module):
    """
    Implement convolutional layer with binary weights and ternary activations, following
    Case 1 of XNOR-Net++.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
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

        self.weight1 = Parameter(conv2d1.weight.detach())
        self.weight2 = Parameter(conv2d2.weight.detach())

    def forward(self, input: Tensor) -> Tensor:
        alpha = (
            (
                (
                    self.weight1.abs().mean((1, 2, 3))
                    + self.weight2.abs().mean((1, 2, 3))
                )
                / 2.0
            )
            .reshape(-1, 1, 1)
            .to(input.device)
        )
        input_sign = Sign.apply(input)
        output1 = F.conv2d(input_sign, Sign.apply(self.weight1), **self.conv2d_params)
        output2 = F.conv2d(input_sign, Sign.apply(self.weight2), **self.conv2d_params)
        return (output1 + output2).mul(alpha)

    # TODO: compute ternary weights from weight_1 and weight_2
