import torch

from torch import Tensor
from torch.nn import Conv2d, Module, Parameter
import torch.nn.functional as F

from .xnorpp import Sign


class Conv2d_XnorPP_SCA(Module):
    """
    Implement convolutional layer with ternary weights and binary activations, following
    SCA and Case 2 of XNOR-Net++.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super(Conv2d_XnorPP_SCA, self).__init__()

        if kwargs["bias"]:
            raise NotImplementedError("bias is not supported on Conv2d_SCA")
        del kwargs["bias"]

        self.conv2d_params = kwargs
        conv2d = Conv2d(
            in_channels, out_channels, kernel_size, bias=False, **self.conv2d_params
        )

        self.weight = Parameter(conv2d.weight.detach())
        # uniform distribution of {-1, 0, 1}
        # nn.init.uniform_(self.weight, a=3*math.atanh(-0.5), b=3*math.atanh(0.5))
        self.alpha = Parameter(torch.ones(out_channels).reshape(-1, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        tanh_weight = torch.tanh(self.weight)
        input = F.conv2d(
            Sign.apply(input),
            tanh_weight if self.training else torch.round(tanh_weight),
            **self.conv2d_params
        )
        return input.mul(self.alpha)

    def wdr(self, alpha: float = 0.1, lamb: float = 1e-7) -> Tensor:
        """
        Implement Weight Decay Regulariser (WDR) for SCA.
        """
        # sparsity = (torch.round(torch.tanh(self.weight)) == 0).sum()
        tanh_weight_sq = torch.tanh(self.weight).square()
        # quant_err = (tanh_weight_sq * (1 - tanh_weight_sq)).sum()
        output = (lamb * (alpha - tanh_weight_sq) * tanh_weight_sq).sum()
        # print(sparsity / self.weight.numel())
        # print(sparsity / self.weight.numel(), quant_err, output)
        return output
