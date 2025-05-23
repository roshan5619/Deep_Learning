# 🧠 CNN-Based Image Classification on MNIST & STL-10

This project demonstrates the use of **Convolutional Neural Networks (CNNs)** to classify images from two popular datasets: **MNIST** (handwritten digits) and **STL-10** (natural scenes). Both datasets contain **10 distinct object classes**, and our goal is to build deep learning models using **Keras or PyTorch** to accurately classify them.

---

## 🎯 Objective

- Train deep CNNs to classify:
  - 🖊️ **MNIST**: 28x28 grayscale images of handwritten digits.
  - 🌄 **STL-10**: 96x96 RGB images from 10 object classes.

- Achieve high accuracy on training and test datasets.
- Provide visual insights into the learning process and model predictions.

---

## 🗃️ Dataset Overview

### 📦 MNIST
- 60,000 training images
- 10,000 test images
- Classes: Digits (0–9)
- Format: 28x28 grayscale

### 🌄 STL-10
- 5,000 training images (500/class)
- 8,000 test images (800/class)
- Classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
- Format: 96x96 RGB

---

## 🧠 Model Architecture (Example)

A typical CNN architecture used for both datasets:

```text
Conv2D → ReLU → MaxPool2D
Conv2D → ReLU → MaxPool2D
Flatten → Dense → Dropout → Dense (Softmax)
```
More complex architectures (e.g., ResNet, VGG) may be used for STL-10 to improve performance.
🔁 Data Preprocessing

    Normalize pixel values to [0, 1] for both datasets.

    One-hot encode labels (if using Keras).

    Resize STL-10 images and apply data augmentation (random crop, horizontal flip).

🧪 Accuracy & Results
Dataset	Model	Train Accuracy	Test Accuracy
MNIST	CNN	~99%	~98.3%
STL-10	CNN	~90%	~80%

    Accuracy may vary slightly based on architecture, training time, and regularization.

📊 Visualizations
1. Sample Predictions

MNIST:

STL-10:

2. Accuracy and Loss Curves

MNIST Accuracy:

STL-10 Accuracy:

📁 Project Structure

📦 CNN_Image_Classification
├── mnist_cnn.py              # CNN training for MNIST
├── stl10_cnn.py              # CNN training for STL-10
├── utils.py                  # Helper functions
├── visuals/                  # Accuracy plots and predictions
│   ├── sample_mnist_preds.png
│   ├── mnist_accuracy.png
│   ├── stl10_accuracy.png
├── README.md                 # Project documentation

🚀 Getting Started
📦 Prerequisites

Install dependencies:

pip install torch torchvision matplotlib numpy

🏃 Run the Code

To train and test on MNIST:

```python mnist_cnn.py
```
To train and test on STL-10:
```
    python stl10_cnn.py
```
Output graphs and predictions will be saved in the visuals/ folder.
🔬 Experiments Performed

    ✅ Compared activation functions: ReLU vs LeakyReLU

    ✅ Experimented with dropout rates

    ✅ Batch sizes: 32, 64, 128

    ✅ Visualized incorrect predictions

    ✅ Used data augmentation on STL-10

👨‍💻 Author

Bandlapalli Roshan Babu
M.Tech AI & DS @ Mahindra University
📧 broshann14@gmail.com
🔗 LinkedIn
🔗 GitHub
📜 License

This project is open-source under the MIT License.
🌟 Acknowledgments

    MNIST Dataset

    STL-10 Dataset

    PyTorch & Keras documentation for model reference

🙌 Fun Fact

Training vision models from scratch teaches patience and the art of tuning. Deep learning isn’t magic—it’s just great math with smart trial and error! 🧠🔥


---

Let me know if you'd like `.md` download support, sample `.py` training code for `mnist_cnn.py`, or a zipped starter project folder with `README`, code, and visuals!

