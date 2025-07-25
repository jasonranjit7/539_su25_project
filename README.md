# MNIST Handwritten Digit Recognition (Team 8)

This project implements a fully connected feedforward neural network to classify handwritten digits from the MNIST dataset. The goal is to deepen our understanding of gradient-based optimization by manually implementing stochastic gradient descent with momentum using PyTorch.

Rather than relying on built-in optimizers like Adam, we explore the fundamental mechanics of parameter updates through direct control of gradients. Despite its simplicity, our model achieves over **97% test accuracy**, demonstrating the effectiveness of manual optimization on a well-structured dataset.

---

## GitHub Repository

[https://github.com/jasonranjit7/539\_su25\_project](https://github.com/jasonranjit7/539_su25_project)

---

## Model Overview

* **Input**: 784-dimensional vector (flattened 28×28 grayscale image)
* **Architecture**:

  * Hidden Layer 1: 100 neurons + ReLU
  * Hidden Layer 2: 50 neurons + ReLU
  * Output Layer: 10 neurons (digits 0–9)
* **Loss Function**: CrossEntropyLoss
* **Optimizer**: Manual SGD with momentum

  * `velocity = momentum * velocity + (1 - momentum) * grad`
  * `param -= lr * velocity`
* **Training Configuration**:

  * Epochs: 10
  * Batch Size: 10
  * Learning Rate: 0.01
  * Momentum: 0.9
* **Data Preprocessing**: Pixel values normalized with mean = 0.5, std = 0.5

---

## Project Structure

```
mnist-digit-recognition/
├── models/
│   ├── __init__.py
│   └── mlp.py              # Feedforward neural network definition
├── src/
│   ├── train.py            # Manual SGD training loop
│   ├── evaluate.py         # Accuracy, classification report, confusion matrix
│   └── visualize.py        # Sample predictions from test set
├── results/
│   └── confusion_matrix.png
├── mnist_mlp.pth           # Saved trained model
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/jasonranjit7/539_su25_project.git
cd 539_su25_project
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Project

```bash
PYTHONPATH=. python3 src/train.py
PYTHONPATH=. python3 src/evaluate.py
PYTHONPATH=. python3 src/visualize.py
```

---

## References

* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE.
* MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
* Kaggle MNIST Dataset: [https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)