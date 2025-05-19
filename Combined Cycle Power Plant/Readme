# 🔥 Combined Cycle Power Plant Energy Output Prediction using ANN (No Deep Learning Libraries)

This repository presents a **from-scratch implementation of an Artificial Neural Network (ANN)** to perform **regression** on the [Combined Cycle Power Plant dataset](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant) from the UCI Machine Learning Repository. 

This project avoids high-level deep learning libraries (like TensorFlow or PyTorch), and instead uses only **NumPy**, **Matplotlib**, and basic Python to build a complete training and inference pipeline — demonstrating how complex nonlinear relationships can be captured purely from data.

---

## 🎯 Objective

Predict the net hourly electrical energy output (**PE**, in MW) of a Combined Cycle Power Plant using four environmental input features:
- **AT**: Ambient Temperature
- **V**: Exhaust Vacuum
- **AP**: Ambient Pressure
- **RH**: Relative Humidity

---

## 📊 Dataset

- 📦 **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
- 📈 **Size**: ~9,568 samples
- 🧮 **Inputs**: 4 numeric features
- 🎯 **Output**: 1 target variable (PE - energy output)

---

## ⚙️ Features

### 🔁 Data Preprocessing
- Normalization of inputs/outputs in the range **[-0.9, +0.9]**
- Output normalized to [0, 1] when using logistic activation
- Virtual min/max for robust normalization (5% outside actual range)

### 🧠 ANN Architecture
- Flexible architecture: 1+ hidden layers, user-defined neurons
- Custom activation functions: `tanh`, `sigmoid`, `ReLU`
- Adjustable hyperparameters: learning rate, batch size, λ (L2), β (momentum)

### 🔧 Training Engine
- Implements **Forward + Backward Propagation** from scratch
- Mini-batch gradient descent with shuffled data
- Supports **L2 regularization** and **momentum-based updates**
- Training + Validation loss tracking at every epoch

### 🧪 Evaluation
- Testing uses saved weights from best validation epoch
- Error metric: **Mean Absolute Percentage Error (MAPE)** on validation and test sets

---

## 📁 File Structure
📦 PowerPlant_ANN_FromScratch
├── ann_model.py # Core ANN engine
├── utils.py # Normalization, MAPE, split utilities
├── train.py # Full training workflow
├── test.py # Evaluation using saved weights
├── weights/ # Directory to save learned weights
├── data/ # Original and normalized datasets
├── plots/ # Training/validation loss plots
└── README.md # Project description


---

## 🧪 Experiments & Hyperparameters

| Parameter       | Values Tested                         |
|----------------|----------------------------------------|
| Activation Fn   | tanh, sigmoid, ReLU                   |
| Batch Sizes     | 1 (SGD), 64, 256, full batch          |
| Learning Rate   | 0.0001, 0.001, 0.01                   |
| L2 Regularization (λ) | 0, 0.1, 0.5, 0.95                |
| Momentum (β)    | 0.9                                   |

Each experiment compares convergence histories and MAPE on validation/test datasets to understand performance impacts.

---

## 📉 Output

At the end of training:
- Save learned weights to file
- Plot **training vs. validation loss**
- Evaluate on test data using **forward-only mode**

---

## 📈 Sample Plots

![Loss Plot](plots/loss_curve_example.png)

---

## 🚀 How to Run

1. Clone the repository:
       git clone https://github.com/your-username/powerplant-ann-fromscratch.git
       cd powerplant-ann-fromscratch 
2.Install dependencies:
 
      pip install numpy matplotlib

3.Run training:

      python train.py

4.Run testing (using saved weights):
  
      python test.py
