# deeplense_gsoc
<br>
<h2>Gravitational Lens Detection using Deep Learning<h2/>
This project implements a machine learning pipeline to identify gravitational lenses from astronomical images using PyTorch.
The model learns to classify whether an input image contains a strong gravitational lens or a non-lensed galaxy.
The dataset contains images captured in three astronomical filters, stored as NumPy arrays with shape:
(3, 64, 64)
<br><hr>
Each sample belongs to one of two classes:
Lens – galaxy images containing gravitational lensing structures such as arcs or rings
Non-Lens – normal galaxy images without lensing
<br><hr>
The model is trained and evaluated using ROC curve and AUC score.
<br><hr>
Project Pipeline
1.Data Loading<br>
2.Load .npy images from dataset folders<br>
3.Assign labels (Lens = 1, Non-Lens = 0)<br>
4.Data Preprocessing<br>
5.Normalize pixel values<br>
6.Convert NumPy arrays to PyTorch tensors<br>
7.Model Architecture<br>
8.Convolutional Neural Network (CNN) or transfer learning model (ResNet18)<br>
9.Training<br>
10.Binary Cross Entropy loss<br>
11.Adam optimizer<br>
12.Mini-batch training with PyTorch DataLoader<br>
13.Evaluation<br>
14.ROC Curve<br>
15.AUC score<br>
<br><hr>
Required libraries:
Python <br>
PyTorch<br>
NumPy<br>
Matplotlib<br>
scikit-learn<br>

<br><hr>
Typical performance for a baseline CNN:
AUC ≈ 0.85 – 0.92
Using transfer learning (ResNet18) and data augmentation can further improve performance.
<br>
Future Improvements:
Data augmentation (rotations and flips)<br>
Transfer learning with pretrained CNNs<br>
Vision Transformer models<br>
Handling class imbalance using weighted loss<br>
<br>
Author:  <b>Sayed Ammar</b>
