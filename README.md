# deeplense_gsoc
\n
\n
Gravitational Lens Detection using Deep Learning-->
This project implements a machine learning pipeline to identify gravitational lenses from astronomical images using PyTorch.
The model learns to classify whether an input image contains a strong gravitational lens or a non-lensed galaxy.
The dataset contains images captured in three astronomical filters, stored as NumPy arrays with shape:
(3, 64, 64)
\n
Each sample belongs to one of two classes:
Lens – galaxy images containing gravitational lensing structures such as arcs or rings
Non-Lens – normal galaxy images without lensing
\n
The model is trained and evaluated using ROC curve and AUC score.
\n
Project Pipeline
1.Data Loading\n
2.Load .npy images from dataset folders\n
3.Assign labels (Lens = 1, Non-Lens = 0)\n
4.Data Preprocessing\n
5.Normalize pixel values\n
6.Convert NumPy arrays to PyTorch tensors\n
7.Model Architecture\n
8.Convolutional Neural Network (CNN) or transfer learning model (ResNet18)\n
9.Training\n
10.Binary Cross Entropy loss\n
11.Adam optimizer\n
12.Mini-batch training with PyTorch DataLoader\n
13.Evaluation\n
14.ROC Curve\n
15.AUC score\n
\n
\n
Required libraries:
Python \n
PyTorch\n
NumPy\n
Matplotlib\n
scikit-learn\n

\n
\n
Typical performance for a baseline CNN:
AUC ≈ 0.85 – 0.92
Using transfer learning (ResNet18) and data augmentation can further improve performance.
\n
Future Improvements:
Data augmentation (rotations and flips)\n
Transfer learning with pretrained CNNs\n
Vision Transformer models\n
Handling class imbalance using weighted loss\n
\n
Author:  Sayed Ammar
