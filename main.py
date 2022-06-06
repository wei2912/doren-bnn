from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
import torch.nn as nn
from torchinfo import summary
from sklearn.metrics import top_k_accuracy_score
from tqdm import trange
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch

import argparse
from pathlib import Path
import time

from doren_bnn.mobilenet import MobileNet, NetType

parser = argparse.ArgumentParser(description="doren_bnn experiments")
parser.add_argument(
    "--num-epochs", default=80, type=int, help="number of epochs to run"
)
parser.add_argument("-b", "--batch-size", default=256, type=int, help="mini-batch size")
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
    val_loader = DataLoader(
        val_set, sampler=RandomSampler(val_set, num_samples=500), **loader_params
    )

    nettype = NetType(kwargs["nettype"])
    model = nn.Sequential(MobileNet(3, 224, num_classes=10, nettype=nettype)).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, 25, gamma=0.1)

    summary(model, input_size=(batch_size, 3, 224, 224))

    if not kwargs["resume"]:
        last_epoch = -1
    else:
        cp = load_checkpoint(cp_path, model, optimizer, scheduler)
        last_epoch = cp["epoch"]

    num_epochs = kwargs["num_epochs"]
    for epoch in trange(last_epoch + 1, num_epochs):
        train(train_loader, writer, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, writer, model, criterion, epoch)

        writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)
        writer.flush()
        scheduler.step()

        save_checkpoint(cp_path, model, optimizer, scheduler, val_loss, epoch)


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
    model,
    criterion,
    optimizer,
    epoch: int,
):
    model.train()

    start = time.monotonic()

    losses = []
    for (input, target) in train_loader:
        output = model(input.cuda())
        loss = criterion(output, target.cuda())

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
    val_loader: DataLoader, writer: SummaryWriter, model, criterion, epoch: int
):
    model.eval()

    losses = []
    outputs = []
    targets = []
    for (input, target) in val_loader:
        output = model(input.cuda())
        loss = criterion(output, target.cuda())

        losses.append(loss.item())
        outputs.extend(output.squeeze().tolist())
        targets.extend(target.tolist())

    loss_mean = sum(losses) / len(losses)
    writer.add_scalar("Test/loss", loss_mean, epoch)
    writer.add_scalar("Test/top-1", top_k_accuracy_score(targets, outputs, k=1), epoch)
    writer.add_scalar("Test/top-5", top_k_accuracy_score(targets, outputs, k=5), epoch)
    writer.flush()

    return loss_mean


if __name__ == "__main__":
    args = parser.parse_args()

    main(**vars(args))
