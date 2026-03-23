# 🔭 DeepLense — Gravitational Lens Detection

<div align="center">

![GSoC](https://img.shields.io/badge/Google_Summer_of_Code-2025-orange?style=flat&logo=google)
![ML4SCI](https://img.shields.io/badge/ML4SCI-DeepLense-blueviolet?style=flat)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**Detecting strong gravitational lenses in wide-field astronomical surveys using deep learning.**

*GSoC 2025 Evaluation Task — ML4SCI DeepLense | Specific Test II: Lens Finding*
*Author: Sayed Ammar*

</div>

---

## 📌 Table of Contents

- [What is Gravitational Lensing?](#-what-is-gravitational-lensing)
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)

---

## 🌌 What is Gravitational Lensing?

When a **massive galaxy or galaxy cluster** sits between us and a distant light source, its gravity bends the background light — like a glass lens. This creates **arcs, rings, or multiple images** of the background galaxy in the sky.

```
   Background Galaxy        Massive Galaxy (Lens)       Observer (Us)
         ★          →→→→      [  GALAXY  ]      →→→→        👁
                               ↑ bends light ↑
                        Creates arcs/rings in image
```

Finding these lenses is important for studying **dark matter**, measuring the **Hubble constant**, and testing **Einstein's theory of general relativity**.

> ⚠️ **The Core Problem:** In large surveys like HSC-SSP, there are millions of galaxy images but only a few hundred are actual lenses. This extreme class imbalance makes automated detection very difficult.

---

## 🚀 Project Overview

This project builds a full end-to-end deep learning pipeline to:

1. Load and explore real astronomical images — 3-filter, 64×64 pixel `.npy` files
2. Analyse the class imbalance problem in the dataset
3. Build and train a ResNet18 transfer learning model
4. Evaluate using Accuracy, ROC Curve, and AUC Score

---

## 📊 Dataset

### Image Format

| Property       | Value                                     |
|----------------|-------------------------------------------|
| Shape          | `(3, 64, 64)`                             |
| Channels       | 3 astronomical filters (g, r, i bands)   |
| Pixel values   | Float32, normalised to `[0, 1]`           |
| Storage format | `.npy` (NumPy binary file)                |
| Task type      | Binary classification (Lens / Non-Lens)  |

### Class Distribution — Actual Dataset

| Split        | Class        | Label | Count  | Percentage |
|--------------|--------------|-------|--------|------------|
| **Training** | Lens         | `1`   | 1,730  | 5.7%       |
| **Training** | Non-Lens     | `0`   | 28,675 | 94.3%      |
| **Test**     | Lens         | `1`   | 195    | 1.0%       |
| **Test**     | Non-Lens     | `0`   | 19,455 | 99.0%      |

```
Training Set — Class Imbalance
────────────────────────────────────────────────────────────
Lens        ██                                         1,730
Non-Lens    ████████████████████████████████████████  28,675
────────────────────────────────────────────────────────────
            Class imbalance ratio  =  16.6 : 1  (Non-Lens : Lens)
```

> **Why does this matter?** A model that always predicts "Non-Lens" would get 94% accuracy but would never find a single lens! This is why **AUC and ROC curves** are used as the main metrics — not accuracy.

### Dataset Folder Structure

```
dataset/
├── train_lenses/        ←  1,730  .npy files  (Lens — training)
├── train_nonlenses/     ← 28,675  .npy files  (Non-Lens — training)
├── test_lenses/         ←    195  .npy files  (Lens — test)
└── test_nonlenses/      ← 19,455  .npy files  (Non-Lens — test)
```

---

## 🔄 Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DeepLense Pipeline                            │
│                                                                      │
│  Step 1           Step 2              Step 3            Step 4      │
│  ─────────────    ──────────────      ──────────────    ──────────  │
│  Load .npy    →   Normalise      →   ResNet18      →   Evaluate    │
│  files            pixels             Transfer           ROC + AUC   │
│  Assign           Convert to         Learning                       │
│  Labels           PyTorch tensors    Model                          │
│  (1 or 0)         shape(B,3,64,64)   BCE Loss                       │
│                                      Adam Optimizer                 │
└──────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step

| Step | Name                   | Details                                                        |
|------|------------------------|----------------------------------------------------------------|
| 1    | Mount & Extract        | Load dataset from Google Drive, extract zip file              |
| 2    | Explore & Visualise    | Plot lens/non-lens images, count class sizes, check imbalance |
| 3    | Custom Dataset Class   | `LensDataset` inherits from PyTorch `Dataset`                 |
| 4    | Normalisation          | `(img - min) / (max - min)` per image → range `[0, 1]`       |
| 5    | DataLoader             | Batch size = 32, shuffled training loader                     |
| 6    | Model Setup            | ResNet18 pretrained, all layers frozen, FC head replaced      |
| 7    | Training               | 10 epochs, `BCEWithLogitsLoss`, Adam (lr = 0.001)             |
| 8    | Evaluation             | Accuracy, ROC Curve, AUC Score on unseen test set             |

---

## 🧠 Model Architecture

### ResNet18 with Transfer Learning

All ResNet18 backbone layers are **frozen** (not updated during training). Only the final classification head is trained — this is called **feature extraction**.

```
Input Image  (3, 64, 64)
        │
        ▼
┌──────────────────────────────────────┐
│   ResNet18 Backbone  — FROZEN ❄️     │
│                                      │
│   Conv1  (3 → 64, 7×7, stride 2)    │
│   BatchNorm → ReLU → MaxPool         │
│                                      │
│   Layer1: 2× BasicBlock  (64 ch)    │
│   Layer2: 2× BasicBlock  (128 ch)   │
│   Layer3: 2× BasicBlock  (256 ch)   │
│   Layer4: 2× BasicBlock  (512 ch)   │
│                                      │
│   AdaptiveAvgPool → feature (512,)  │
└──────────────────┬───────────────────┘
                   │  512-dim feature vector
                   ▼
┌──────────────────────────────────────┐
│   Classification Head — TRAINABLE 🔥 │
│                                      │
│   Linear(512 → 1)                    │
│   BCEWithLogitsLoss (sigmoid inside) │
└──────────────────┬───────────────────┘
                   │
                   ▼
         P(Lens) — score in [0, 1]
```

### Layer-by-Layer Shape

| Layer                | Output Shape      | Notes                          |
|----------------------|-------------------|--------------------------------|
| Input                | `(B, 3, 64, 64)`  | 3-channel astronomical image   |
| Conv1 + BN + ReLU    | `(B, 64, 32, 32)` | 7×7 kernel, stride 2           |
| MaxPool              | `(B, 64, 16, 16)` | 3×3, stride 2                  |
| Layer1 (2 blocks)    | `(B, 64, 16, 16)` | No downsampling                |
| Layer2 (2 blocks)    | `(B, 128, 8, 8)`  | Stride 2 downsampling          |
| Layer3 (2 blocks)    | `(B, 256, 4, 4)`  | Stride 2 downsampling          |
| Layer4 (2 blocks)    | `(B, 512, 2, 2)`  | Stride 2 downsampling          |
| AdaptiveAvgPool      | `(B, 512, 1, 1)`  | Global average pooling         |
| Flatten              | `(B, 512)`        | Feature vector                 |
| FC Head (Linear)     | `(B, 1)`          | Final classification score     |

### Why Freeze the Backbone?

| Strategy                 | What is Trained        | Best For                           |
|--------------------------|------------------------|------------------------------------|
| Freeze all (this repo)   | FC head only           | Small dataset, fast training ✅    |
| Fine-tune last layer     | FC + Layer4            | Medium dataset, better accuracy    |
| Fine-tune all layers     | Everything             | Large dataset, maximum accuracy    |

---

## 📉 Training

### Loss Per Epoch — Actual Output

| Epoch | Training Loss | Change         |
|-------|---------------|----------------|
| 0     | 0.17532       | —              |
| 1     | 0.15636       | ↓ −0.01896     |
| 2     | 0.15352       | ↓ −0.00284     |
| 3     | 0.15306       | ↓ −0.00046     |
| 4     | 0.15192       | ↓ −0.00114     |
| 5     | 0.14978       | ↓ −0.00214 ← best |
| 6     | 0.15147       | ↑ +0.00169     |
| 7     | 0.15281       | ↑ +0.00134     |
| 8     | 0.15104       | ↓ −0.00177     |
| 9     | 0.15184       | ↑ +0.00080     |

```
Training Loss Curve
──────────────────────────────────────────────────
0.175 │▓
0.165 │ ▓
0.156 │  ▓▓
0.152 │    ▓▓▓
0.150 │       ▓▓▓▓▓▓▓
──────────────────────────────────────────────────
       0   1   2   3   4   5   6   7   8   9
                        Epoch
```

**Observation:** Loss drops sharply in the first 2 epochs then stabilises around 0.151–0.153. This is expected because only the tiny FC head is being trained.

### Training Configuration

| Hyperparameter    | Value                |
|-------------------|----------------------|
| Epochs            | 10                   |
| Batch size        | 32                   |
| Optimizer         | Adam                 |
| Learning rate     | 0.001                |
| Loss function     | `BCEWithLogitsLoss`  |
| Trainable layers  | FC head only         |
| Device            | GPU (CUDA) / CPU     |

---

## 📈 Results

### Final Performance — Actual Numbers

| Metric            | Value      | Notes                                              |
|-------------------|------------|----------------------------------------------------|
| **Test Accuracy** | **97.68%** | High due to class imbalance — see warning below   |
| **AUC Score**     | **0.8815** | Main metric — honest measure of discrimination    |
| Threshold used    | 0.5        | Default Sigmoid threshold                          |
| Test samples      | 19,650     | (195 lens + 19,455 non-lens)                       |

> ⚠️ **Important — why accuracy is misleading here:** The test set has ~99% non-lenses. A model that always predicts "Non-Lens" would get 99% accuracy — yet it would never find a single real lens! **AUC = 0.88 is the honest metric** that shows the model genuinely separates lenses from non-lenses.

### ROC Curve — Explanation

```
  True Positive Rate (Recall)
  1.0 │           ╭──────────────────────
      │        ╭──╯   ResNet18  AUC = 0.88
  0.8 │      ╭─╯
      │    ╭─╯
  0.6 │   ╭╯
      │  ╭╯
  0.4 │ ╭╯
      │╱   ← random baseline (AUC = 0.50)
  0.0 └────────────────────────────────
      0.0  0.2  0.4  0.6  0.8  1.0
            False Positive Rate
```

### AUC Score Reference Table

| AUC Score       | Interpretation                  |
|-----------------|---------------------------------|
| 1.00            | Perfect model                   |
| 0.90 – 0.99     | Excellent                       |
| **0.88**        | **Good — our result ✅**        |
| 0.70 – 0.85     | Acceptable                      |
| 0.50            | Random guessing (no skill)      |

---

## ⚙️ Installation

```bash
# 1. Clone this repository
git clone https://github.com/ammar01-me/DeepLense.git
cd DeepLense

# 2. Install all required libraries
pip install torch torchvision numpy matplotlib scikit-learn
```

### Required Libraries

| Library        | Version  | Purpose                                       |
|----------------|----------|-----------------------------------------------|
| `torch`        | ≥ 2.0    | Model building, training, and inference       |
| `torchvision`  | ≥ 0.15   | Pretrained ResNet18 model weights             |
| `numpy`        | ≥ 1.23   | Loading `.npy` image files                    |
| `matplotlib`   | ≥ 3.6    | Plotting images, loss curves, ROC curve       |
| `scikit-learn` | ≥ 1.2    | Computing ROC curve and AUC score             |

---

## 🧪 Usage

### 1. Prepare the dataset (Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
zip_path = "/content/drive/MyDrive/gravitational_lense/lens-finding-test.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/dataset")
print("Dataset extracted!")
```

### 2. Check class sizes

```python
import os

train_lenses    = len(os.listdir("dataset/train_lenses"))      # 1730
train_nonlenses = len(os.listdir("dataset/train_nonlenses"))   # 28675
ratio = train_nonlenses / train_lenses
print(f"Imbalance ratio: {ratio:.2f} non-lens per lens")
# Output: 16.58 non-lens per lens
```

### 3. Train the model

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LensDataset(Dataset):
    def __init__(self, lens_dir, nonlens_dir):
        self.images, self.labels = [], []
        for f in os.listdir(lens_dir):
            self.images.append(os.path.join(lens_dir, f))
            self.labels.append(1)
        for f in os.listdir(nonlens_dir):
            self.images.append(os.path.join(nonlens_dir, f))
            self.labels.append(0)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = np.load(self.images[idx])
        img = (img - img.min()) / (img.max() - img.min())
        return torch.tensor(img, dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

train_dataset = LensDataset("dataset/train_lenses", "dataset/train_nonlenses")
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Build model
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False          # freeze backbone
model.fc = nn.Linear(model.fc.in_features, 1)  # replace head
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer  = optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images).squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.5f}")
```

### 4. Evaluate — ROC curve and AUC

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_true, y_scores = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        probs = torch.sigmoid(model(images.to(device)))
        y_true.extend(labels.numpy())
        y_scores.extend(probs.cpu().numpy())

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")   # AUC: 0.8815

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='tomato', lw=2, label=f'ResNet18 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random baseline (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Gravitational Lens Detection')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=150)
plt.show()
```

---

## 🔮 Future Improvements

| Improvement                          | Expected Benefit                                         | Status      |
|--------------------------------------|----------------------------------------------------------|-------------|
| Data augmentation (flips, rotations) | Reduces overfitting, improves generalisation            | 🔜 Planned  |
| Weighted loss / Focal loss           | Better handling of 16:1 class imbalance                 | 🔜 Planned  |
| Fine-tune more ResNet layers         | Higher AUC — backbone adapts to astronomy images        | 🔜 Planned  |
| EfficientNet / ViT backbone          | State-of-the-art accuracy on image tasks                | 🔜 Planned  |
| Threshold tuning for max Recall      | Catch more real lenses even at cost of false positives  | 🔜 Planned  |
| Precision-Recall curve               | Fairer evaluation for highly imbalanced data            | 🔜 Planned  |
| GradCAM visualisation                | Show which part of the image the model focuses on       | 🔜 Planned  |
| Apply to real HSC-SSP sky data       | Discover actual new gravitational lens candidates       | 🎯 Goal     |
| Cross-survey evaluation (DES, LSST) | Check if model generalises to other telescopes          | 🎯 Goal     |

---

## 📚 References

- [DeepLense — ML4SCI GSoC 2025](https://ml4sci.org/gsoc/projects/2025/project_DEEPLENSE.html)
- [HSC-SSP Wide-Field Survey](https://hsc-release.mtk.nao.ac.jp/)
- He et al. (2016) — Deep Residual Learning for Image Recognition (ResNet paper)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

<div align="center">

Made with ❤️ for GSoC 2025 — ML4SCI DeepLense
**Author: Sayed Ammar**

</div>
