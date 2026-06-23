# Cats vs Dogs Image Classification (PyTorch CNN)

## ⚠️ NO API KEYS OR LOGIN CREDENTIALS ARE TO EVER BE STORED IN THIS REPOSITORY ⚠️

This repository contains a Convolutional Neural Network (CNN) image classification system built using PyTorch.

The project classifies images of cats and dogs using a custom neural network architecture trained from scratch. The workflow includes exploratory data analysis (EDA), image preprocessing, dataset validation, model training, performance evaluation, and visualization of results.

The project was developed as a practical introduction to computer vision, deep learning, GPU acceleration, and the PyTorch framework.

---

# High-Level Overview

At a high level, the system performs the following steps:

1. Loads image datasets from training and testing folders
2. Performs exploratory data analysis (EDA) on the images
3. Validates image quality and searches for any corrupted files
4. Standardizes image dimensions and color channels (RGB/Greyscale)
5. Converts images into PyTorch tensors
6. Builds a custom CNN architecture from scratch
7. Trains the model using GPU acceleration (CUDA)
8. Evaluates model performance on unseen test data
9. Generates a confusion matrix and classification metrics
10. Supports experimentation with model architecture and hyperparameters

---

# Project Structure

```
Image-Recognition-Cats-and-Dogs/
│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   │
│   └── test/
│       ├── cats/
│       └── dogs/
│
├── .gitignore
├── README.md
```

# Dataset

Dataset Source:

* Kaggle Cats and Dogs Image Classification Dataset:
https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification

Dataset Structure:

```text
train/
├── cats/
└── dogs/

test/
├── cats/
└── dogs/
```

Training Set:

* 279 Cat Images
* 278 Dog Images

Testing Set:

* 70 Cat Images
* 70 Dog Images

Total Images:

* 697

---

# Exploratory Data Analysis (EDA)

The notebook performs several validation and quality assurance steps before training:

## Class Balance Verification

* Confirms equal representation of cats and dogs
* Detects potential class imbalance issues

## Image Validation

* Searches for corrupted image files
* Verifies image readability using Pillow (PIL)

## Color Channel Analysis

* Identifies RGB and non-RGB images
* Automatically converts grayscale images when necessary

## Image Dimension Analysis

Measures:

* Minimum Width
* Maximum Width
* Mean Width
* Median Width
* Minimum Height
* Maximum Height
* Mean Height
* Median Height

Visualization includes:

* Width distributions
* Height distributions
* Sample image inspection

---

# Data Preprocessing

Prior to training, the following transformations are applied:

## Image Standardization

* Resize all images to 128 × 128 pixels

## Color Conversion

* Convert all images to RGB format

## Tensor Conversion

* Convert images into PyTorch tensor objects

## Normalization

Pixel values are normalized to improve training stability and convergence.

## Data Augmentation

Training images may undergo:

* Horizontal flipping
* Rotation experiments
* Additional augmentation testing

---

# Machine Learning Model

## Framework

* PyTorch
* TorchVision

## Hardware Acceleration

* NVIDIA CUDA
* GTX 1080 GPU

## Architecture

Custom CNN built from scratch:

```
Input Image (128 × 128 × 3)
        │
        ▼
Convolution Layer (3 → 16)
        │
        ▼
ReLU Activation
        │
        ▼
Max Pooling
        │
        ▼
Convolution Layer (16 → 32)
        │
        ▼
ReLU Activation
        │
        ▼
Max Pooling
        │
        ▼
Flatten
        │
        ▼
Fully Connected Layer
        │
        ▼
Dropout
        │
        ▼
Output Layer (Cat / Dog)
```

# Training Configuration

## Optimizer

* Adam Optimizer

## Loss Function

* CrossEntropyLoss

## Learning Rate

```
0.0005
```

## Batch Size

```
32
```

## Epochs

Typically:

```
10 - 20
```

depending on experimentation and model performance.

---

# Model Evaluation

Performance is evaluated using:

## Accuracy

Measures:

```
Correct Predictions / Total Predictions
```

## Confusion Matrix

Tracks:

* True Positives
* True Negatives
* False Positives
* False Negatives

## Classification Performance

Used to determine:

* Cat recognition performance
* Dog recognition performance
* Model bias toward a specific class

---

# Results

Current experimentation has achieved approximately:

```
~70% Test Accuracy
```

using a custom CNN trained entirely from scratch.

This serves as a baseline model for future experimentation and improvement.

---

# Future Improvements

Potential future enhancements include:

## Model Architecture

* Additional convolution layers
* Batch normalization
* Residual connections

## Training Improvements

* Early stopping
* Validation datasets
* Learning-rate scheduling

## Data Improvements

* Larger datasets
* Additional augmentation techniques

## Transfer Learning

Potential experimentation with:

* ResNet
* EfficientNet
* MobileNet
* VGG

---

# Educational Objectives

This project was developed to gain hands-on experience with:

* Deep Learning
* Computer Vision
* Convolutional Neural Networks (CNNs)
* PyTorch
* GPU Acceleration (CUDA)
* Image Classification Workflows
* Neural Network Evaluation and Tuning

---

# Configuration & Security

## All Sensitive Information Is Excluded From Version Control

Examples include:

* API Keys
* Database Credentials
* Personal Access Tokens
* Environment Files

---

## ⚠️ NO API KEYS OR LOGIN CREDENTIALS ARE TO EVER BE STORED IN THIS REPOSITORY ⚠️