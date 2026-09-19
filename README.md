# ResNet18 CIFAR-10 Classification with Two-Neuron Bottleneck Analysis

This project explores the impact of a two-neuron bottleneck layer on a pre-trained ResNet18 model fine-tuned for CIFAR-10 image classification. It compares the performance of a standard classification head against a specialized two-neuron bottleneck head, and visualizes the learned feature representations and the optimization landscape of the bottleneck weights.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Generated Files](#generated-files)

## Project Overview

This repository contains code and analysis for classifying CIFAR-10 images using a ResNet18 model. The key focus is on understanding how reducing the dimensionality of the features before the final classification layer (a 'bottleneck') affects performance and interpretability. We train two types of classification heads on top of a frozen ResNet18 backbone:

1.  **Baseline Head**: A single `Linear` layer (512 features -> 10 classes).
2.  **Two-Neuron Head**: A bottleneck structure with two `Linear` layers (512 features -> 2 features -> 10 classes).

The project also includes visualization of the 2D feature space learned by the two-neuron bottleneck and an examination of the loss landscape for a pair of weights during gradient descent.

## Setup and Installation

To run this project, you need Python and PyTorch. Other required libraries are listed below and can be installed via `pip`:

```bash
pip install torch torchvision matplotlib numpy tqdm pandas-gbq koreanize-matplotlib
```

## Data

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is used, consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

Images are transformed to 64x64 pixels, converted to tensors, and normalized using ImageNet's mean and standard deviation: `MEAN=(0.485, 0.456, 0.406)` and `STD=(0.229, 0.224, 0.225)`.

## Model Architecture

The core of the model is a `resnet18` backbone, pre-trained on ImageNet. The final classification layer of ResNet18 is replaced to fit the 10 classes of CIFAR-10.

Two different classification 'heads' are trained on top of the *frozen* ResNet18 backbone's 512-dimensional features:

