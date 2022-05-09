from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from torch import optim, nn
from torchinfo import summary
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="doren_bnn experiments")
parser.add_argument("data_str", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--num-epochs", default=90, type=int, help="number of epochs to run"
)
parser.add_argument(
    "-b", "--batch-size", default=256, type=int, help="mini-batch size (default: 256)"
)


def main(**kwargs):
    data_path = Path(kwargs["data_str"])
    train_path = data_path / "train"
    test_path = data_path / "test"

    writer = SummaryWriter()

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
    trainset = Subset(
        CIFAR10(train_path, train=True, transform=transform, download=True), range(512)
    )
    testset = Subset(
        CIFAR10(test_path, train=False, transform=transform, download=True), range(512)
    )

    batch_size = kwargs["batch_size"]
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(mobilenet_v3_small(pretrained=True), nn.Linear(1000, 10))
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-4)

    summary(model, input_size=(batch_size, 3, 224, 224))

    num_epochs = kwargs["num_epochs"]
    for epoch in tqdm(range(num_epochs)):
        train(trainloader, writer, model, criterion, optimiser, epoch)
        test(testloader, writer, model, criterion, epoch)

    test(testloader, writer, model, criterion, num_epochs)


def train(
    trainloader: DataLoader,
    writer: SummaryWriter,
    model,
    criterion,
    optimiser,
    epoch: int,
):
    model.train()

    losses = []
    for (input, target) in trainloader:
        output = model(input)
        loss = criterion(output, target)

        losses.append(loss.item())

        optimiser.zero_grad()
        loss.backward()

        optimiser.step()

    loss_mean = sum(losses) / len(losses)
    writer.add_scalar("Train/loss", loss_mean, epoch)
    writer.flush()


def test(testloader: DataLoader, writer: SummaryWriter, model, criterion, epoch: int):
    model.eval()

    losses = []
    outputs = []
    targets = []
    for (input, target) in testloader:
        output = model(input)
        loss = criterion(output, target)

        losses.append(loss.item())
        outputs.extend(output.squeeze().tolist())
        targets.extend(target.tolist())

    loss_mean = sum(losses) / len(losses)
    writer.add_scalar("Test/loss", loss_mean, epoch)
    writer.add_scalar("Test/top-1", top_k_accuracy_score(targets, outputs, k=1), epoch)
    writer.add_scalar("Test/top-5", top_k_accuracy_score(targets, outputs, k=5), epoch)
    writer.flush()


if __name__ == "__main__":
    args = parser.parse_args()

    main(**vars(args))
