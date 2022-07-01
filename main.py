from torchinfo import summary
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch

import argparse

# from doren_bnn.mobilenet import MobileNet, NetType

from doren_bnn_concrete import preload_keys
from doren_bnn.mobilenet import NetType
from doren_bnn.toynet import ToyNet, ToyNet_FHE
from doren_bnn.utils import Dataset, Experiment

parser = argparse.ArgumentParser(description="doren_bnn experiments")
parser.add_argument(
    "--num-epochs", default=120, type=int, help="number of epochs to run"
)
parser.add_argument("-b", "--batch-size", default=32, type=int, help="mini-batch size")
parser.add_argument("--id", nargs="?", type=str, help="experiment id")
parser.add_argument(
    "--resume",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="resume from latest checkpoint?",
)
parser.add_argument(
    "--nettype",
    default=NetType.REAL,
    choices=[x.value for x in NetType._member_map_.values()],
    help="type of network",
)


def main(**kwargs):
    nettype = NetType(kwargs["nettype"])
    num_epochs = kwargs["num_epochs"]
    batch_size = kwargs["batch_size"]

    print(nettype)  # FIXME

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = MobileNet(3, num_classes=10, nettype=nettype).to(device)
    NUM_INPUT = 10
    model = ToyNet(num_input=NUM_INPUT, num_classes=10).to(device)
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=5e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 30)

    summary(model, input_size=(batch_size, 3, 224, 224))

    experiment = Experiment(kwargs["id"], Dataset.CIFAR10, batch_size)
    experiment.train(
        device,
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        resume=kwargs["resume"],
    )

    # Test FHE version of model

    preload_keys()

    model_fhe = ToyNet_FHE(num_input=NUM_INPUT, num_classes=10)
    experiment.load_checkpoint(model_fhe, optimizer, scheduler)
    experiment.test(device, model)
    experiment.test_fhe(model_fhe)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
