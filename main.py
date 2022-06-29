from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
import torch.nn as nn
from torchinfo import summary
from sklearn.metrics import top_k_accuracy_score
from tqdm import trange
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch

import argparse
from pathlib import Path
import time

from doren_bnn_concrete import preload_keys

# from doren_bnn.mobilenet import MobileNet, NetType

from doren_bnn.mobilenet import NetType
from doren_bnn.toynet import ToyNet, ToyNet_FHE

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
    data_path = Path("data/")
    train_path = data_path / "train"
    val_path = data_path / "val"

    runs_path = Path("runs/")
    if kwargs["id"] is None:
        writer = SummaryWriter()
        run_path = Path(writer.get_logdir())
    else:
        run_path = runs_path / kwargs["id"]
        writer = SummaryWriter(log_dir=run_path)
    cp_path = run_path / "checkpoint.pt"

    # Refer to https://pytorch.org/vision/stable/models.html for more details on the
    # normalisation of torchvision's datasets.
    transform = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_params = {"transform": transform, "download": True}
    train_set = CIFAR10(train_path, train=True, **dataset_params)
    val_set = CIFAR10(val_path, train=False, **dataset_params)

    batch_size = kwargs["batch_size"]
    loader_params = {"batch_size": batch_size, "pin_memory": True}
    train_loader = DataLoader(
        train_set, sampler=RandomSampler(train_set, num_samples=2500), **loader_params
    )
    val_loader = DataLoader(Subset(val_set, range(500)), **loader_params)
    test_loader = DataLoader(Subset(val_set, range(2)), **loader_params)

    nettype = NetType(kwargs["nettype"])
    num_epochs = kwargs["num_epochs"]

    print(nettype)  # FIXME

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = MobileNet(3, num_classes=10, nettype=nettype).cuda()
    NUM_INPUT = 10
    model = ToyNet(num_input=NUM_INPUT, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=5e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 30)

    summary(model, input_size=(batch_size, 3, 224, 224))

    if not kwargs["resume"]:
        last_epoch = -1
    else:
        cp = load_checkpoint(cp_path, model, optimizer, scheduler)
        last_epoch = cp["epoch"]

    for epoch in trange(last_epoch + 1, num_epochs):
        train(train_loader, writer, device, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, writer, device, model, criterion, epoch)

        writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)
        writer.flush()
        scheduler.step()

        save_checkpoint(cp_path, model, optimizer, scheduler, val_loss, epoch)

    # Test FHE version of model

    preload_keys()
    model_fhe = ToyNet_FHE(num_input=NUM_INPUT, num_classes=10)
    cp = load_checkpoint(cp_path, model_fhe, optimizer, scheduler)
    test(test_loader, writer, device, model)
    test_fhe(test_loader, writer, model_fhe)


def save_checkpoint(path, model, optimizer, scheduler, val_loss: float, epoch: int):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    cp = torch.load(path)
    model.load_state_dict(cp["model_state_dict"])
    optimizer.load_state_dict(cp["optimizer_state_dict"])
    scheduler.load_state_dict(cp["scheduler_state_dict"])
    return cp


def train(
    train_loader: DataLoader,
    writer: SummaryWriter,
    device,
    model,
    criterion,
    optimizer,
    epoch: int,
):
    model.train()
    start = time.monotonic()

    losses = []
    for (input, target) in train_loader:
        output = model(input.to(device))
        # loss = criterion(output, target.to(device))
        loss = criterion(output, target.to(device)) + model.wdr()

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    end = time.monotonic()
    writer.add_scalar("Train/time", end - start, epoch)

    loss_mean = sum(losses) / len(losses)
    writer.add_scalar("Train/loss", loss_mean, epoch)
    writer.flush()


def validate(
    val_loader: DataLoader, writer: SummaryWriter, device, model, criterion, epoch: int
):
    model.eval()
    start = time.monotonic()

    losses = []
    outputs = []
    targets = []
    for (input, target) in val_loader:
        output = model(input.to(device))
        # loss = criterion(output, target.to(device))
        loss = criterion(output, target.to(device)) + model.wdr()

        losses.append(loss.item())
        outputs.extend(output.squeeze().tolist())
        targets.extend(target.tolist())

    end = time.monotonic()
    writer.add_scalar("Val/time", end - start, epoch)

    loss_mean = sum(losses) / len(losses)
    writer.add_scalar("Val/loss", loss_mean, epoch)
    writer.add_scalar(
        "Val/top-1",
        top_k_accuracy_score(targets, outputs, k=1, labels=range(10)),
        epoch,
    )
    writer.add_scalar(
        "Val/top-5",
        top_k_accuracy_score(targets, outputs, k=5, labels=range(10)),
        epoch,
    )
    writer.flush()

    return loss_mean


def test(test_loader: DataLoader, writer: SummaryWriter, device, model):
    model.eval()
    start = time.monotonic()

    outputs = []
    targets = []
    for (input, target) in test_loader:
        output = model(input.to(device))
        outputs.extend(output.squeeze().tolist())
        targets.extend(target.tolist())
        print(output[:10])

    end = time.monotonic()
    writer.add_scalar("Test/time", end - start, -1)

    writer.add_scalar(
        "Test/top-1", top_k_accuracy_score(targets, outputs, k=1, labels=range(10)), -1
    )
    writer.add_scalar(
        "Test/top-5", top_k_accuracy_score(targets, outputs, k=5, labels=range(10)), -1
    )
    writer.flush()


def test_fhe(test_loader: DataLoader, writer: SummaryWriter, model):
    model.eval()
    start = time.monotonic()

    outputs = []
    targets = []
    for (input, target) in test_loader:
        output = model(input)
        outputs.extend(output.squeeze().tolist())
        targets.extend(target.tolist())
        print(output[:10])

    end = time.monotonic()
    writer.add_scalar("Test-FHE/time", end - start, -1)

    writer.add_scalar(
        "Test-FHE/top-1",
        top_k_accuracy_score(targets, outputs, k=1, labels=range(10)),
        -1,
    )
    writer.add_scalar(
        "Test-FHE/top-5",
        top_k_accuracy_score(targets, outputs, k=5, labels=range(10)),
        -1,
    )
    writer.flush()


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
