# MNIST Digit Classification (Team 8)

This project explores handwritten digit recognition using the MNIST dataset. We implement a fully connected neural network (MLP) with a **manually implemented SGD optimizer with momentum**, and compare it with **PyTorch’s built-in optimizers** and **CNN architectures** to evaluate performance, convergence, and generalizability.

---

## Team Members

- **Jason Ranjit Joseph Rajasekar**  
  ECE Department, Graduate Student  
  josephrajase@wisc.edu

- **Hae Seung Pyun**  
  Computer Science, Undergraduate  
  hpyun2@wisc.edu

- **Emma Vigy**  
  Computer Science, Undergraduate  
  evigy@wisc.edu

---

## Project Structure

```
mnist-digit-recognition/
├── models/
│   ├── mlp.py              # MLP architecture
│   └── cnn.py              # CNN architecture
├── src/
│   ├── train_mlp_manual.py # Manual SGD with momentum
│   ├── train_mlp_builtin.py# PyTorch optimizer (MLP)
│   ├── train_cnn.py        # CNN + PyTorch optimizer
│   └── evaluate.py         # Confusion matrix, predictions
├── results/
│   ├── accuracy_plots.png
│   ├── confusion_matrix.png
│   └── mlp_manual.pth      # Saved model weights
├── requirements.txt
└── README.md
```

---

## Objective

Our goals:
- Implement **manual SGD with momentum** from scratch.
- Train a basic MLP on MNIST using this optimizer.
- Compare performance with:
  - PyTorch's built-in SGD and Adam
  - A simple convolutional neural network (CNN)
- Visualize accuracy, loss, and confusion matrices.
- Evaluate generalization across models.

---

## Models

### 1. MLP (Fully Connected)
- Input: 784 (28×28) flattened image  
- Hidden Layers: 100 → 50 neurons (ReLU)  
- Output: 10 classes (digits 0–9)

### 2. CNN
- Conv → ReLU → Pool → Conv → ReLU → Pool → FC → FC  
- Standard CNN architecture for digit recognition.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/jasonranjit7/539_su25_project.git
cd mnist-digit-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Models

```bash
python src/train_mlp_manual.py         # Manual SGD
python src/train_mlp_builtin.py        # PyTorch SGD
python src/train_cnn.py                # CNN with Adam
```

### 4. Evaluate

```bash
python src/evaluate.py
```

---

## 🧺 Results Summary

| Model              | Optimizer     | Test Accuracy |
|-------------------|---------------|---------------|
| MLP (manual SGD)  | Manual SGD    | ~98.2%        |
| MLP (PyTorch SGD) | PyTorch SGD   | ~97.1%        |
| CNN               | PyTorch Adam  | ~99.2%        |

See:
- `results/accuracy_plots.png` for training curves  
- `results/confusion_matrix.png` for evaluation metrics  

---

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
- [MNIST Dataset – Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

---
