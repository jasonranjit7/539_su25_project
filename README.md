# Image & Digit Classification (Team 8)

## Overview

This project explores **handwritten digit and image classification** using both **manual** and **built-in** optimizers across multiple datasets (MNIST, Fashion-MNIST, CIFAR-10).
The primary goal was to deepen our understanding of optimization algorithms by implementing **Stochastic Gradient Descent (SGD) with momentum** from scratch, then comparing it against **PyTorch’s built-in SGD and Adam optimizers**.

We also tested **different architectures** (MLP, CNN) and **datasets** to evaluate optimizer generalizability and performance.

---

## Features

* **Manual SGD with momentum** implementation
* Comparison against:

  * PyTorch’s **SGD with momentum**
  * PyTorch’s **Adam** and **AdamW**
* **Multiple architectures**:

  * Fully Connected MLP
  * Convolutional Neural Network (CNN)
* Evaluation on:

  * MNIST (digits 0–9)
  * Fashion-MNIST (clothing images)
  * CIFAR-10 (color images)
* Visualization:

  * Accuracy and loss curves
  * Confusion matrices
  * Comparison plots across models & optimizers

---

## Project Structure

```
Final_Project/
│── models/
│   ├── mlp.py                  # MLP architecture
│   ├── cnn.py                  # CNN architecture
│
│── src/
│   ├── optim_manual.py         # Manual SGD with momentum
│   ├── utils.py                # Helper functions
│   ├── train_template.py       # Shared training loop
│   ├── train_mlp_manual.py     # MLP with manual SGD
│   ├── train_mlp_builtin.py    # MLP with built-in SGD
│   ├── train_mlp_fashion.py    # Fashion-MNIST MLP training
│   ├── train_cnn_manual_vs_builtin.py  # CNN manual vs built-in
│   ├── train_cnn_fashion.py    # Fashion-MNIST CNN training
│   ├── train_cnn_cifar10.py    # CIFAR-10 CNN training
│   ├── evaluate_all.py         # Evaluation & plots for MNIST runs
│   ├── evaluate_extra_datasets.py  # Evaluation for Fashion-MNIST & CIFAR-10
│
│── requirements.txt
│── README.md                  
```

---

## Installation

```bash
git clone https://github.com/jasonranjit7/539_su25_project.git
cd Final_Project

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Code

### **MNIST – MLP**

```bash
PYTHONPATH=. python src/train_mlp_manual.py
PYTHONPATH=. python src/train_mlp_builtin.py
```

### **MNIST – CNN**

```bash
PYTHONPATH=. python src/train_cnn_manual_vs_builtin.py
```

### **Fashion-MNIST**

```bash
PYTHONPATH=. python src/train_mlp_fashion.py
PYTHONPATH=. python src/train_cnn_fashion.py
```

### **CIFAR-10**

```bash
PYTHONPATH=. python src/train_cnn_cifar10.py
```

### **Evaluation & Plots**

```bash
PYTHONPATH=. python src/evaluate_all.py
PYTHONPATH=. python src/evaluate_extra_datasets.py
```

---

## Outputs

* **Training logs**: Printed to console during runs
* **Saved models**: `results/model_name.pth`
* **Plots**: Saved in `results/` folder

  * Loss curves
  * Accuracy curves
  * Confusion matrices
  * Model comparison bar charts

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324. [https://doi.org/10.1109/5.726791](https://doi.org/10.1109/5.726791)
2. MNIST Database. (n.d.). Yann LeCun's MNIST handwritten digit database. [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
3. PyTorch Documentation. (n.d.). [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
4. Deng, L. (2012). *The MNIST database of handwritten digit images for machine learning research.* IEEE Signal Processing Magazine, 29(6), 141–142. [https://doi.org/10.1109/MSP.2012.2211477](https://doi.org/10.1109/MSP.2012.2211477)
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)