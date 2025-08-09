import os, glob, torch as t
from models.mlp import MLP
from models.cnn import SimpleCNN
from src.utils import get_dataloaders, get_device, evaluate_model, save_confusion, ensure_dir

def infer_arch_from_path(p):
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None, None, None
    dataset = parts[0]
    arch = parts[1]
    optimizer = "_".join(parts[2:])
    return dataset, arch, optimizer

def build_model(arch, in_channels, input_dim):
    if arch == "mlp":
        return MLP(input_dim=input_dim)
    elif arch == "cnn":
        return SimpleCNN(in_channels=in_channels)
    else:
        return None

def main():
    device = get_device()
    ensure_dir("results/plots")

    for pth in glob.glob("results/*.pth"):
        dataset, arch, optimizer = infer_arch_from_path(pth)
        if dataset is None:
            continue
        print("Evaluating:", pth)
        loaders = get_dataloaders(dataset=dataset, batch_size=128, grayscale_cifar=True)
        train_loader, test_loader, in_channels, input_dim = loaders
        model = build_model(arch, in_channels, input_dim)
        if model is None:
            continue
        model.load_state_dict(t.load(pth, map_location=device))
        model.to(device)
        acc, loss, y, yhat = evaluate_model(model, test_loader, device)
        out_cm = f"results/{dataset}_{arch}_{optimizer}_cm_reval.png"
        save_confusion(y, yhat, labels=list(range(10)), out_path=out_cm)
        print(f"Re-evaluated {pth} | acc={acc:.4f} | saved {out_cm}")

if __name__ == "__main__":
    main()