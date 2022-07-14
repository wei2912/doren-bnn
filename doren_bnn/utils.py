import torch

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from sklearn.metrics import top_k_accuracy_score
from tqdm.auto import trange

from collections.abc import Callable
from enum import Enum
from pathlib import Path
import time
import math


class Dataset(Enum):
    CIFAR10 = "CIFAR10"


class Experiment:
    def __init__(
        self, id: str | None, dataset: Dataset, batch_size: int, multiplier: float = 1.0
    ):
        torch.manual_seed(0)  # ensure train/val split is reproducible

        DATA_PATH = Path("data/")
        TRAIN_PATH = DATA_PATH / "train"
        VAL_PATH = DATA_PATH / "val"

        RUNS_PATH = Path("runs/")
        if id is None:
            self._writer = SummaryWriter()
            RUN_PATH = Path(self._writer.get_logdir())
        else:
            RUN_PATH = RUNS_PATH / id
            self._writer = SummaryWriter(log_dir=RUN_PATH)
        self._CP_PATH = RUN_PATH / "checkpoint.pt"

        self._last_epoch = -1

        dataset_params = {
            # Refer to https://pytorch.org/vision/stable/models.html for more details on
            # the normalisation of torchvision's datasets.
            "transform": Compose(
                [
                    Resize(32),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
            "download": True,
        }
        if dataset == Dataset.CIFAR10:
            self._NUM_CLASSES = 10
            train_set = CIFAR10(TRAIN_PATH, train=True, **dataset_params)
            val_set = CIFAR10(VAL_PATH, train=False, **dataset_params)
        else:
            raise NotImplementedError(f"dataset {dataset} is not available")

        loader_params = {"batch_size": batch_size, "pin_memory": True}

        num_train_samples = math.floor(len(train_set) * multiplier)
        self._train_loader = DataLoader(
            Subset(train_set, torch.randperm(len(train_set))[:num_train_samples]),
            **loader_params,
        )

        num_val_samples = math.floor(len(val_set) * multiplier)
        self._val_loader = DataLoader(
            Subset(val_set, torch.randperm(len(val_set))[:num_val_samples]),
            **loader_params,
        )
        self._test_loader = DataLoader(
            Subset(val_set, torch.randperm(len(val_set))[:num_val_samples]),
            **loader_params,
        )

    def _add_scalars(self, mode: str, scalars_dict: dict[str, float], epoch: int = -1):
        for scalar_name, scalar_val in scalars_dict.items():
            self._writer.add_scalar(f"{mode}/{scalar_name}", scalar_val, epoch)
        self._writer.flush()

    def save_checkpoint(self, model, optimizer, scheduler, epoch: int):
        self._last_epoch = epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            self._CP_PATH,
        )

    def load_checkpoint(self, model, optimizer, scheduler):
        cp = torch.load(self._CP_PATH)
        self._last_epoch = cp["epoch"]
        model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        scheduler.load_state_dict(cp["scheduler_state_dict"])
        return cp

    def train(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs: int,
        hyperparams_dict: dict[str, Callable[[int], float]] = None,
        **kwargs,
    ):
        for epoch in trange(
            self._last_epoch + 1,
            num_epochs,
            initial=self._last_epoch + 1,
            total=num_epochs,
        ):
            hyperparams_val_dict = {
                name: f(epoch) for name, f in hyperparams_dict.items()
            }
            kwargs = {**kwargs, **hyperparams_val_dict}

            self.train_epoch(model, criterion, optimizer, epoch, **kwargs)
            self._add_scalars(
                "Train",
                {
                    "lr": scheduler.get_last_lr()[0],
                    **hyperparams_val_dict,
                },
                epoch,
            )
            scheduler.step()

            self.validate_epoch(model, criterion, epoch, **kwargs)

            self.save_checkpoint(model, optimizer, scheduler, epoch=epoch)

    def train_epoch(
        self,
        model,
        criterion,
        optimizer,
        epoch: int,
        device=None,
        regulariser: Callable[..., torch.TensorType] = None,
        **kwargs,
    ):
        model.train()
        start = time.monotonic()

        losses = []
        outputs = []
        targets = []
        for (input, target) in self._train_loader:
            if device is not None:
                input = input.to(device)
                target = target.to(device)

            output = model(input)
            loss = (
                criterion(output, target)
                if regulariser is None
                else criterion(output, target)
                + regulariser(**{"model": model, **kwargs})
            )

            losses.append(loss.item())
            outputs.extend(output.squeeze().tolist())
            targets.extend(target.tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = time.monotonic()

        self._add_scalars(
            "Train",
            {
                "time": end - start,
                "loss": sum(losses) / len(losses),
                "top-1": top_k_accuracy_score(
                    targets, outputs, k=1, labels=range(self._NUM_CLASSES)
                ),
                "top-5": top_k_accuracy_score(
                    targets, outputs, k=5, labels=range(self._NUM_CLASSES)
                ),
            },
            epoch=epoch,
        )

    def validate_epoch(
        self,
        model,
        criterion,
        epoch: int,
        device=None,
        regulariser: Callable[..., torch.TensorType] = None,
        **kwargs,
    ):
        model.eval()
        start = time.monotonic()

        losses = []
        outputs = []
        targets = []
        for (input, target) in self._val_loader:
            if device is not None:
                input = input.to(device)
                target = target.to(device)

            output = model(input)
            loss = (
                criterion(output, target)
                if regulariser is None
                else criterion(output, target)
                + regulariser(**{"model": model, **kwargs})
            )

            losses.append(loss.item())
            outputs.extend(output.squeeze().tolist())
            targets.extend(target.tolist())

        end = time.monotonic()
        self._add_scalars(
            "Val",
            {
                "time": end - start,
                "loss": sum(losses) / len(losses),
                "top-1": top_k_accuracy_score(
                    targets, outputs, k=1, labels=range(self._NUM_CLASSES)
                ),
                "top-5": top_k_accuracy_score(
                    targets, outputs, k=5, labels=range(self._NUM_CLASSES)
                ),
            },
            epoch=epoch,
        )

        return sum(losses) / len(losses)

    def test(self, model, device=None):
        model.eval()
        start = time.monotonic()

        outputs = []
        targets = []
        for (input, target) in self._test_loader:
            if device is not None:
                input = input.to(device)
            output = model(input)
            outputs.extend(output.squeeze().tolist())
            targets.extend(target.tolist())

        end = time.monotonic()
        self._add_scalars(
            "Test",
            {
                "time": end - start,
                "top-1": top_k_accuracy_score(
                    targets, outputs, k=1, labels=range(self._NUM_CLASSES)
                ),
                "top-5": top_k_accuracy_score(
                    targets, outputs, k=5, labels=range(self._NUM_CLASSES)
                ),
            },
        )

    def test_fhe(self, model_fhe):
        model_fhe.eval()
        start = time.monotonic()

        outputs = []
        targets = []
        for (input, target) in self._test_loader:
            output = model_fhe(input)
            outputs.extend(output.squeeze().tolist())
            targets.extend(target.tolist())

        end = time.monotonic()
        self._add_scalars(
            "Test-FHE",
            {
                "time": end - start,
                "top-1": top_k_accuracy_score(
                    targets, outputs, k=1, labels=range(self._NUM_CLASSES)
                ),
                "top-5": top_k_accuracy_score(
                    targets, outputs, k=5, labels=range(self._NUM_CLASSES)
                ),
            },
        )
