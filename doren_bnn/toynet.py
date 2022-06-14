from torch import Tensor
from torch.nn import (
    Module,
    Linear,
)
import torch.nn.functional as F
from tqdm import tqdm

from .xnorpp import Sign

from doren_bnn_concrete import preload_keys, toynet


class ToyNet(Module):
    def __init__(self, num_classes: int = 1000, **kwargs):
        super(ToyNet, self).__init__()

        self.fc = Linear(8 * 8 * 3, num_classes, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        input = input.view(-1, 8 * 8 * 3)
        return F.linear(Sign.apply(input), Sign.apply(self.fc.weight))


class ToyNet_FHE(ToyNet):
    def __init__(self, **kwargs):
        super(ToyNet_FHE, self).__init__(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        assert not self.training

        state_dict = self.state_dict()
        state_dict["fc.weight"] = [
            [w > 0 for w in row] for row in self.fc.weight.tolist()
        ]

        preload_keys()

        input = input.view(-1, 8 * 8 * 3).tolist()
        output = []
        for im in tqdm(input):
            output.append(toynet(state_dict, im))
        return Tensor(output)