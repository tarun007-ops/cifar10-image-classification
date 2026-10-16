"""Train and evaluate the CIFAR-10 CNN used in this portfolio project."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


class CIFAR10CNN(nn.Module):
    """Three convolution blocks followed by a compact classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return loss_sum / total, 100.0 * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            loss_sum += loss_fn(logits, labels).item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size
            encoded = labels.cpu() * 10 + predictions.cpu()
            confusion += torch.bincount(encoded, minlength=100).reshape(10, 10)

    return loss_sum / total, 100.0 * correct / total, confusion


def save_training_plot(losses: list[float], accuracies: list[float], path: Path) -> None:
    epochs = range(1, len(losses) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, losses, marker="o", color="#31688e")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, accuracies, marker="o", color="#2e7d32")
    axes[1].set(title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xticks(list(epochs))
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_confusion_plot(matrix: torch.Tensor, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    axis.set(
        xticks=range(10),
        yticks=range(10),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="CIFAR-10 Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max().item() / 2
    for row in range(10):
        for column in range(10):
            value = matrix[row, column].item()
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "#202124",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Images")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch size must be positive; workers cannot be negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__}")
    print(f"Training device: {device}")

    transform = transforms.ToTensor()
    train_data = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    print(f"Training images: {len(train_data)} | Test images: {len(test_data)}")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    model = CIFAR10CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    accuracies: list[float] = []

    for epoch in range(1, args.epochs + 1):
        loss, accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        losses.append(loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")

    test_loss, test_accuracy, confusion = evaluate(model, test_loader, loss_fn, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "cifar10_cnn.pth")
    save_training_plot(losses, accuracies, args.output_dir / "training_history.png")
    save_confusion_plot(confusion, args.output_dir / "confusion_matrix.png")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Saved model and result plots in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
