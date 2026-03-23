# DeepLense — Gravitational Lens Detection

<div align="center">
   
**Detecting strong gravitational lenses in wide-field astronomical surveys using deep learning.**

*GSoC 2025 Evaluation Task — ML4SCI DeepLense | Lens Finding*
*Author: Sayed Ammar*

</div>

---

## Table of Contents

- [What is Gravitational Lensing?](#-what-is-gravitational-lensing)
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Results](#-results)
- [Future Improvements](#-future-improvements)

---

## What is Gravitational Lensing?

When a **massive galaxy or galaxy cluster** sits between us and a distant light source, its gravity bends the background light — like a glass lens. This creates **arcs, rings, or multiple images** of the background galaxy in the sky.

```
   Background Galaxy        Massive Galaxy (Lens)       Observer (Us)
         ★          →→→→      [  GALAXY  ]      →→→→        👁
                               ↑ bends light ↑
                        Creates arcs/rings in image
```

Finding these lenses is important for studying **dark matter**, measuring the **Hubble constant**, and testing **Einstein's theory of general relativity**.

> **The Core Problem:** In large surveys like HSC-SSP, there are millions of galaxy images but only a few hundred are actual lenses. This extreme class imbalance makes automated detection very difficult.

---

## Project Overview

This project builds a full end-to-end deep learning pipeline to:

1. Load and explore real astronomical images — 3-filter, 64×64 pixel `.npy` files
2. Analyse the class imbalance problem in the dataset
3. Build and train a ResNet18 transfer learning model
4. Evaluate using Accuracy, ROC Curve, and AUC Score

---

## Dataset

### Image Format

| Property       | Value                                     |
|----------------|-------------------------------------------|
| Shape          | `(3, 64, 64)`                             |
| Channels       | 3 astronomical filters (g, r, i bands)    |
| Pixel values   | Float32, normalised to `[0, 1]`           |
| Storage format | `.npy` (NumPy binary file)                |
| Task type      | Binary classification (Lens / Non-Lens)   |

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

## Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DeepLense Pipeline                            │
│                                                                      │
│  Step 1           Step 2              Step 3            Step 4       │
│  ─────────────    ──────────────      ──────────────    ──────────   │
│  Load .npy    →   Normalise      →   ResNet18      →   Evaluate      │
│  files            pixels             Transfer           ROC + AUC    │
│  Assign           Convert to         Learning                        │
│  Labels           PyTorch tensors    Model                           │
│  (1 or 0)         shape(B,3,64,64)   BCE Loss                        │
│                                      Adam Optimizer                  │
└──────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step

| Step | Name                   | Details                                                        |
|------|------------------------|----------------------------------------------------------------|
| 1    | Mount & Extract        | Load dataset from Google Drive, extract zip file               |
| 2    | Explore & Visualise    | Plot lens/non-lens images, count class sizes, check imbalance  |
| 3    | Custom Dataset Class   | `LensDataset` inherits from PyTorch `Dataset`                  |
| 4    | Normalisation          | `(img - min) / (max - min)` per image → range `[0, 1]`         |
| 5    | DataLoader             | Batch size = 32, shuffled training loader                      |
| 6    | Model Setup            | ResNet18 pretrained, all layers frozen, FC head replaced       |
| 7    | Training               | 10 epochs, `BCEWithLogitsLoss`, Adam (lr = 0.001)              |
| 8    | Evaluation             | Accuracy, ROC Curve, AUC Score on unseen test set              |

---

## Model Architecture

### ResNet18 with Transfer Learning

All ResNet18 backbone layers are **frozen** (not updated during training). Only the final classification head is trained — this is called **feature extraction**.

```
Input Image  (3, 64, 64)
        │
        ▼
┌──────────────────────────────────────┐
│   ResNet18 Backbone  — FROZEN        │
│                                      │
│   Conv1  (3 → 64, 7×7, stride 2)     │
│   BatchNorm → ReLU → MaxPool         │
│                                      │
│   Layer1: 2× BasicBlock  (64 ch)     │
│   Layer2: 2× BasicBlock  (128 ch)    │
│   Layer3: 2× BasicBlock  (256 ch)    │
│   Layer4: 2× BasicBlock  (512 ch)    │
│                                      │
│   AdaptiveAvgPool → feature (512,)   │
└──────────────────┬───────────────────┘
                   │  512-dim feature vector
                   ▼
┌──────────────────────────────────────┐
│   Classification Head — TRAINABLE    │
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
| Freeze all (this repo)   | FC head only           | Small dataset, fast training       |
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
| Test Accuracy     | 97.68%     | High due to class imbalance — see warning below    |
| AUC Score         | 0.8815     | Main metric — honest measure of discrimination     |
| Threshold used    | 0.5        | Default Sigmoid threshold                          |
| Test samples      | 19,650     | (195 lens + 19,455 non-lens)                       |

>  **Important — why accuracy is misleading here:** The test set has ~99% non-lenses. A model that always predicts "Non-Lens" would get 99% accuracy — yet it would never find a single real lens! **AUC = 0.88 is the honest metric** that shows the model genuinely separates lenses from non-lenses.


### AUC Score Reference Table

| AUC Score       | Interpretation                  |
|-----------------|---------------------------------|
| 1.00            | Perfect model                   |
| 0.90 – 0.99     | Excellent                       |
| 0.88            | Good — our result               |
| 0.70 – 0.85     | Acceptable                      |
| 0.50            | Random guessing (no skill)      |

---

## Future Improvements

| Improvement                          | Expected Benefit                                         | Status    |
|--------------------------------------|----------------------------------------------------------|-----------|
| Data augmentation (flips, rotations) | Reduces overfitting, improves generalisation             |  Planned  |
| Weighted loss / Focal loss           | Better handling of 16:1 class imbalance                  |  Planned  |
| Fine-tune more ResNet layers         | Higher AUC — backbone adapts to astronomy images         |  Planned  |
| EfficientNet / ViT backbone          | State-of-the-art accuracy on image tasks                 |  Planned  |
| Threshold tuning for max Recall      | Catch more real lenses even at cost of false positives   |  Planned  |
| Precision-Recall curve               | Fairer evaluation for highly imbalanced data             |  Planned  |
| GradCAM visualisation                | Show which part of the image the model focuses on        |  Planned  |
| Apply to real HSC-SSP sky data       | Discover actual new gravitational lens candidates        |  Goal     |
| Cross-survey evaluation (DES, LSST)  | Check if model generalises to other telescopes           |  Goal     |

---

## References

- [DeepLense — ML4SCI GSoC 2025](https://ml4sci.org/gsoc/projects/2025/project_DEEPLENSE.html)
- [HSC-SSP Wide-Field Survey](https://hsc-release.mtk.nao.ac.jp/)
- He et al. (2016) — Deep Residual Learning for Image Recognition (ResNet paper)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

<div align="center">
**Author: Sayed Ammar**

</div>
