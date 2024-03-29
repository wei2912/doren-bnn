import torch

from torch import Tensor
from torch.nn import Conv2d, Module, Parameter
from torch.autograd.function import Function
import torch.nn.functional as F


class Sign(Function):
    r"""
    Implement forward and backward pass for the sign function, with the Hard-Tanh
    function as the backward pass.

    For an input $r$,
    Forward pass: $\text{sign}(r)$, where $\text{sign}(r) = 1$ if $r > 0$, and $-1$
    otherwise.
    Backward pass: $\frac{\del \text{sign}}{\del r} = r1_{|r| \leq 1}$
    """

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        ctx.save_for_backward(input)
        return 2.0 * input.gt(0.0) - 1.0

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (input,) = ctx.saved_tensors
        return grad_output.clone() * input.le(1.0) * input.ge(-1.0)


class Conv2d_XnorPP(Module):
    """
    Implement convolutional layer with binary weights and activations, following Case 1
    of XNOR-Net++.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super(Conv2d_XnorPP, self).__init__()

        if kwargs["bias"]:
            raise NotImplementedError("bias is not supported on Conv2d_XnorPP")
        del kwargs["bias"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv2d_params = kwargs
        conv2d = Conv2d(
            in_channels, out_channels, kernel_size, bias=False, **self.conv2d_params
        )

        self.weight = Parameter(conv2d.weight)
        self.alpha = Parameter(torch.ones(out_channels).reshape(-1, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        output = F.conv2d(
            Sign.apply(input), Sign.apply(self.weight), **self.conv2d_params
        )
        return output.mul(self.alpha)
