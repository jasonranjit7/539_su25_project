import os, json, glob
import matplotlib.pyplot as plt
from src.utils import ensure_dir

def load_all_metrics(pattern="results/*_all_metrics.json"):
    files = glob.glob(pattern)
    runs = []
    for f in files:
        with open(f, "r") as fh:
            data = json.load(fh)
            dataset_arch = os.path.basename(f).replace("_all_metrics.json", "")
            runs.append((dataset_arch, data["runs"]))
    return runs

def main():
    ensure_dir("results/plots")
    runs = load_all_metrics()

    for dataset_arch, entries in runs:
        labels = [e["optimizer"] for e in entries]
        test_accs = [e["test_acc"] for e in entries]

        plt.figure(figsize=(8,4))
        plt.bar(labels, test_accs)
        plt.ylabel("Test Accuracy")
        plt.xticks(rotation=30, ha="right")
        plt.title(f"Accuracy by Optimizer: {dataset_arch}")
        plt.tight_layout()
        out = f"results/plots/{dataset_arch}_accuracy_bar.png"
        plt.savefig(out)
        plt.close()
        print("Saved:", out)

if __name__ == "__main__":
    main()