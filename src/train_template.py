import os
import argparse
import torch as t
import torch.optim as optim
from models.mlp import MLP
from models.cnn import SimpleCNN
from src.optim_manual import ManualSGD, ManualAdam, ManualAdamW
from src.utils import (
    set_seed, get_device, get_dataloaders, evaluate_model,
    plot_curves, save_confusion, save_metrics_json, ensure_dir
)

def get_model(arch, in_channels, input_dim, dropout_p, num_classes=10):
    if arch == "mlp":
        return MLP(input_dim=input_dim, hidden_dims=(100, 50), num_classes=num_classes, dropout_p=dropout_p)
    elif arch == "cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)
    else:
        raise ValueError("Unknown arch")

def get_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    name = name.lower()
    if name == "manual_sgd":
        return ManualSGD(params, lr=lr, momentum=momentum)
    if name == "manual_adam":
        return ManualAdam(params, lr=lr)
    if name == "manual_adamw":
        return ManualAdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum)
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def train_one(model, train_loader, test_loader, device, optimizer_name, lr, momentum,
              weight_decay, epochs, out_prefix):
    loss_fn = t.nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": []}
    for ep in range(epochs):
        model.train()
        total = 0
        correct = 0
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running += loss.item() * y.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_loss = running / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        print(f"[{optimizer_name}] Epoch {ep+1}/{epochs}  loss={train_loss:.4f}  acc={train_acc:.4f}")

    acc, test_loss, y_true, y_pred = evaluate_model(model, test_loader, device)

    plot_curves(history, f"{out_prefix}_{optimizer_name}")
    save_confusion(y_true, y_pred, labels=list(range(10)), out_path=f"{out_prefix}_{optimizer_name}_cm.png")
    metrics = {
        "optimizer": optimizer_name,
        "epochs": epochs,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "train_loss_last": history["train_loss"][-1],
        "train_acc_last": history["train_acc"][-1],
        "test_acc": acc,
        "test_loss": test_loss
    }
    save_metrics_json(metrics, f"{out_prefix}_{optimizer_name}.json")
    print(f"Test acc ({optimizer_name}): {acc:.4f}")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion", "cifar10"])
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--optimizers", type=str, nargs="+",
                        default=["manual_sgd", "sgd", "manual_adam", "adam", "manual_adamw", "adamw"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grayscale_cifar", action="store_true", help="Use 1-channel CIFAR-10")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    train_loader, test_loader, in_channels, input_dim = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        grayscale_cifar=args.grayscale_cifar
    )
    model = get_model(args.arch, in_channels, input_dim, args.dropout).to(device)

    ensure_dir("results/metrics")
    out_prefix = f"results/{args.dataset}_{args.arch}"

    all_metrics = []
    for opt in args.optimizers:
        model = get_model(args.arch, in_channels, input_dim, args.dropout).to(device)
        metrics = train_one(
            model, train_loader, test_loader, device,
            optimizer_name=opt,
            lr=(args.lr if "adam" not in opt else 1e-3),
            momentum=args.momentum,
            weight_decay=(1e-2 if opt in ["manual_adamw", "adamw"] else args.weight_decay),
            epochs=args.epochs,
            out_prefix=out_prefix
        )
        t.save(model.state_dict(), f"{out_prefix}_{opt}.pth")
        all_metrics.append(metrics)

    save_metrics_json({"runs": all_metrics}, f"{out_prefix}_all_metrics.json")

if __name__ == "__main__":
    main()