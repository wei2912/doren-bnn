from torch import Tensor
from torch.nn import (
    Module,
    Linear,
)


class ToyNet(Module):
    def __init__(self, num_classes: int = 1000, **kwargs):
        super(ToyNet, self).__init__()

        self.fc = Linear(224 * 224 * 3, num_classes)

    def forward(self, input: Tensor) -> Tensor:
        input = input.view(-1, 224 * 224 * 3)
        return self.fc(input)
