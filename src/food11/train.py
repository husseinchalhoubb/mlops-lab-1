"""Train a ResNet-18 Food-11 classifier and track it with MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORIES = {
    "mini": PROJECT_ROOT / "data" / "food11_processed_mini",
    "processed": PROJECT_ROOT / "data" / "food11_processed",
}
NUM_CLASSES = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=DATASET_DIRECTORIES,
        default="mini",
        help="Dataset version to use.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")
    if args.lr <= 0:
        parser.error("--lr must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    return args


def make_loader(
    dataset_root: Path,
    split: str,
    transform,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = ImageFolder(dataset_root / split, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)

            total_loss += loss.item() * images.size(0)
            predicted_classes = predictions.argmax(dim=1)
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return average_loss, accuracy


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)
    dataset_root = DATASET_DIRECTORIES[args.dataset]

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_root}. Run 'dvc pull' if necessary."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    train_loader = make_loader(
        dataset_root,
        "training",
        transform,
        args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        dataset_root,
        "validation",
        transform,
        args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        dataset_root,
        "evaluation",
        transform,
        args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "architecture": "resnet18",
                "optimizer": "Adam",
                "device": device.type,
            }
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy = evaluate(
                model, val_loader, criterion, device
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_accuracy={val_accuracy:.4f}"
            )

        _, test_accuracy = evaluate(
            model, test_loader, criterion, device
        )
        mlflow.log_metric("test_accuracy", test_accuracy)

        model = model.to("cpu")
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )

        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")


if __name__ == "__main__":
    main()