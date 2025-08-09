import os
import json
from pathlib import Path
import torch as t
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import random

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)

def get_device():
    return "cuda" if t.cuda.is_available() else "cpu"

def get_dataloaders(dataset="mnist", batch_size=64, grayscale_cifar=True):
    dataset = dataset.lower()
    if dataset == "mnist":
        tfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])
        train_ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
        test_ds  = datasets.MNIST("data", train=False, download=True, transform=tfm)
        in_channels = 1
        input_dim = 28 * 28

    elif dataset == "fashion":
        tfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])
        train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=tfm)
        test_ds  = datasets.FashionMNIST("data", train=False, download=True, transform=tfm)
        in_channels = 1
        input_dim = 28 * 28

    elif dataset == "cifar10":
        if grayscale_cifar:
            tfm = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            in_channels = 1
            input_dim = 32 * 32
        else:
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            in_channels = 3
            input_dim = 3 * 32 * 32

        train_ds = datasets.CIFAR10("data", train=True, download=True, transform=tfm)
        test_ds  = datasets.CIFAR10("data", train=False, download=True, transform=tfm)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, in_channels, input_dim

def save_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def plot_curves(history, out_path_prefix):
    ensure_dir(os.path.dirname(out_path_prefix))
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train")
    if "val_acc" in history:
        plt.plot(history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_acc.png")
    plt.close()

def save_metrics_json(metrics_dict, out_path):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_y = []
    all_pred = []
    loss_fn = t.nn.CrossEntropyLoss()
    total_loss = 0.0
    with t.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * y.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
    avg_loss = total_loss / total
    acc = correct / total
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    return acc, avg_loss, all_y, all_pred