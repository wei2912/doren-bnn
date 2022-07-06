import torch

# from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from sklearn.metrics import top_k_accuracy_score
from tqdm.auto import trange

from enum import Enum
from pathlib import Path
import time


class Dataset(Enum):
    CIFAR10 = "CIFAR10"


class Experiment:
    def __init__(self, id: str | None, dataset: Dataset, batch_size: int):
        torch.manual_seed(0)  # ensure train/val split is reproducible

        self.data_path = Path("data/")
        self.train_path = self.data_path / "train"
        self.val_path = self.data_path / "val"

        self.runs_path = Path("runs/")
        if id is None:
            self.writer = SummaryWriter()
            self.run_path = Path(self.writer.get_logdir())
        else:
            self.run_path = self.runs_path / id
            self.writer = SummaryWriter(log_dir=self.run_path)
        self.cp_path = self.run_path / "checkpoint.pt"

        # Refer to https://pytorch.org/vision/stable/models.html for more details on the
        # normalisation of torchvision's datasets.
        transform = Compose(
            [
                Resize(32),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset_params = {"transform": transform, "download": True}
        if dataset == Dataset.CIFAR10:
            self.train_set = CIFAR10(self.train_path, train=True, **dataset_params)
            self.val_set = CIFAR10(self.val_path, train=False, **dataset_params)

        loader_params = {"batch_size": batch_size, "pin_memory": True}
        self.train_loader = DataLoader(
            self.train_set,
            # Subset(self.train_set, torch.randperm(len(self.train_set))[:5000]),
            **loader_params
        )
        self.val_loader = DataLoader(
            self.val_set,
            # Subset(self.val_set, torch.randperm(len(self.val_set))[:1000]),
            **loader_params
        )
        self.test_loader = DataLoader(
            self.val_set,
            # Subset(self.val_set, torch.randperm(len(self.val_set))[:1000]),
            **loader_params
        )

    def save_checkpoint(self, model, optimizer, scheduler, val_loss: float, epoch: int):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            },
            self.cp_path,
        )

    def load_checkpoint(self, model, optimizer, scheduler):
        cp = torch.load(self.cp_path)
        model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        scheduler.load_state_dict(cp["scheduler_state_dict"])
        return cp

    def train(
        self,
        device,
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs: int,
        resume: bool = False,
        **kwargs
    ):
        if not resume:
            last_epoch = -1
        else:
            cp = self.load_checkpoint(model, optimizer, scheduler)
            last_epoch = cp["epoch"]

        lamb = kwargs["lamb"]
        for epoch in trange(
            last_epoch + 1, num_epochs, initial=last_epoch + 1, total=num_epochs
        ):
            # FIXME: abstract out calculation of lamb into an actual lambda function
            if epoch < 50:
                kwargs["lamb"] = 0
            else:
                kwargs["lamb"] = lamb * (10 ** -((num_epochs - epoch) // 50))
            self.writer.add_scalar("Train/lamb", kwargs["lamb"], epoch)

            self.train_epoch(device, model, criterion, optimizer, epoch, **kwargs)
            val_loss = self.validate_epoch(device, model, criterion, epoch, **kwargs)

            self.writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)
            self.writer.flush()
            scheduler.step()

            self.save_checkpoint(model, optimizer, scheduler, val_loss, epoch)

    def train_epoch(
        self, device, model, criterion, optimizer, epoch: int, alpha: float, lamb: float
    ):
        model.train()
        start = time.monotonic()

        losses = []
        outputs = []
        targets = []
        for (input, target) in self.train_loader:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            # loss = criterion(output, target)
            loss = criterion(output, target) + lamb * model.wdr(alpha)

            losses.append(loss.item())
            outputs.extend(output.squeeze().tolist())
            targets.extend(target.tolist())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        end = time.monotonic()
        self.writer.add_scalar("Train/time", end - start, epoch)

        loss_mean = sum(losses) / len(losses)
        self.writer.add_scalar("Train/loss", loss_mean, epoch)
        self.writer.add_scalar(
            "Train/top-1",
            top_k_accuracy_score(targets, outputs, k=1),
            epoch,
        )
        self.writer.add_scalar(
            "Train/top-5",
            top_k_accuracy_score(targets, outputs, k=5),
            epoch,
        )
        self.writer.flush()

    def validate_epoch(
        self, device, model, criterion, epoch: int, alpha: float, lamb: float
    ):
        model.eval()
        start = time.monotonic()

        losses = []
        outputs = []
        targets = []
        for (input, target) in self.val_loader:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            # loss = criterion(output, target)
            loss = criterion(output, target) + lamb * model.wdr(alpha)

            losses.append(loss.item())
            outputs.extend(output.squeeze().tolist())
            targets.extend(target.tolist())

        end = time.monotonic()
        self.writer.add_scalar("Val/time", end - start, epoch)

        loss_mean = sum(losses) / len(losses)
        self.writer.add_scalar("Val/loss", loss_mean, epoch)
        self.writer.add_scalar(
            "Val/top-1",
            top_k_accuracy_score(targets, outputs, k=1),
            epoch,
        )
        self.writer.add_scalar(
            "Val/top-5",
            top_k_accuracy_score(targets, outputs, k=5),
            epoch,
        )
        self.writer.flush()

        return loss_mean

    def test(self, device, model):
        model.eval()
        start = time.monotonic()

        outputs = []
        targets = []
        for (input, target) in self.test_loader:
            input = input.to(device)
            output = model(input)
            outputs.extend(output.squeeze().tolist())
            targets.extend(target.tolist())
            # print(output[:10])

        end = time.monotonic()
        self.writer.add_scalar("Test/time", end - start, -1)

        # FIXME: see labels
        self.writer.add_scalar(
            "Test/top-1",
            top_k_accuracy_score(targets, outputs, k=1, labels=range(10)),
            -1,
        )
        self.writer.add_scalar(
            "Test/top-5",
            top_k_accuracy_score(targets, outputs, k=5, labels=range(10)),
            -1,
        )
        self.writer.flush()

    def test_fhe(self, model_fhe):
        model_fhe.eval()
        start = time.monotonic()

        outputs = []
        targets = []
        for (input, target) in self.test_loader:
            output = model_fhe(input)
            outputs.extend(output.squeeze().tolist())
            targets.extend(target.tolist())
            # print(output[:10])

        end = time.monotonic()
        self.writer.add_scalar("Test-FHE/time", end - start, -1)

        # FIXME: see labels
        self.writer.add_scalar(
            "Test-FHE/top-1",
            top_k_accuracy_score(targets, outputs, k=1, labels=range(10)),
            -1,
        )
        self.writer.add_scalar(
            "Test-FHE/top-5",
            top_k_accuracy_score(targets, outputs, k=5, labels=range(10)),
            -1,
        )
        self.writer.flush()
