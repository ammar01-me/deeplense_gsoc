# deeplense_gsoc


Gravitational Lens Detection using Deep Learning-->
This project implements a machine learning pipeline to identify gravitational lenses from astronomical images using PyTorch.
The model learns to classify whether an input image contains a strong gravitational lens or a non-lensed galaxy.
The dataset contains images captured in three astronomical filters, stored as NumPy arrays with shape:
(3, 64, 64)

Each sample belongs to one of two classes:
Lens – galaxy images containing gravitational lensing structures such as arcs or rings
Non-Lens – normal galaxy images without lensing

The model is trained and evaluated using ROC curve and AUC score.

Project Pipeline
1.Data Loading
2.Load .npy images from dataset folders
3.Assign labels (Lens = 1, Non-Lens = 0)
4.Data Preprocessing
5.Normalize pixel values
6.Convert NumPy arrays to PyTorch tensors
7.Model Architecture
8.Convolutional Neural Network (CNN) or transfer learning model (ResNet18)
9.Training
10.Binary Cross Entropy loss
11.Adam optimizer
12.Mini-batch training with PyTorch DataLoader
13.Evaluation
14.ROC Curve
15.AUC score


Required libraries:
Python 3
PyTorch
NumPy
Matplotlib
scikit-learn



Typical performance for a baseline CNN:
AUC ≈ 0.85 – 0.92
Using transfer learning (ResNet18) and data augmentation can further improve performance.

Future Improvements:
Data augmentation (rotations and flips)
Transfer learning with pretrained CNNs
Vision Transformer models
Handling class imbalance using weighted loss

Author:  Sayed Ammar
