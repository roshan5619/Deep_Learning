# 🧠 CNN-Based Image Classification on MNIST & STL-10

This project demonstrates the use of **Convolutional Neural Networks (CNNs)** to classify images from two popular datasets: **MNIST** (handwritten digits) and **STL-10** (objects in natural scenes). Both datasets contain **10 distinct object classes**, and our aim is to build and train accurate image classifiers using **Keras or PyTorch**.

---

## 🧾 Objective

- Train deep CNN models to classify:
  - 🖊️ **MNIST**: 28x28 grayscale handwritten digits.
  - 🌄 **STL-10**: 96x96 RGB images of objects in natural scenes.
- Achieve **high accuracy** on both training and test datasets.
- Provide visualizations of:
  - Sample inputs & predictions
  - Accuracy/loss curves
  - Misclassified examples

---

## 🗃️ Dataset Overview

### 📦 MNIST
- 60,000 training images
- 10,000 test images
- 10 classes: 0-9
- Image shape: 28x28 pixels, grayscale

### 🌄 STL-10
- 5,000 training images (500/class)
- 8,000 test images (800/class)
- 10 object classes (e.g., bird, airplane, ship, etc.)
- Image shape: 96x96 pixels, RGB

---

## 🧠 Model Architecture

A basic but powerful **CNN** architecture for both datasets. Example (for MNIST):

```text
Conv2D → ReLU → MaxPool2D
Conv2D → ReLU → MaxPool2D
Flatten → Dense → Dropout → Dense (Softmax)
