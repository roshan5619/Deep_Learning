# 🔢 Neural Network Function Approximation using TensorFlow

This project demonstrates how to approximate the mathematical function:

\[
y = x_1^2 + x_2^2
\]

using a feedforward neural network in TensorFlow. We experiment with different activation functions (Sigmoid and ReLU) in the hidden layers and analyze their performance on multiple datasets.

---

## 📊 Objective

Train a neural network to approximate the function:

\[
y = x_1^2 + x_2^2
\]

where \( x_1 \) and \( x_2 \) are scalar inputs. The model is evaluated using two test datasets of different ranges and distributions.

---

## 📁 Project Structure
```
📦 function-approximation-nn
├── generate_data.py # Scripts to generate training and test datasets
├── model_sigmoid.py # Neural network using sigmoid activations
├── model_relu.py # Neural network using ReLU activations
├── utils.py # Plotting and normalization helpers
├── results/
│ ├── training_surface_sigmoid.png
│ ├── test1_surface_sigmoid.png
│ ├── test2_surface_sigmoid.png
│ ├── training_surface_relu.png
│ ├── test1_surface_relu.png
│ ├── test2_surface_relu.png
├── README.md

```
---

## 🧪 Dataset Description

### 🟢 Training Dataset (483 samples)

- Cartesian product of:
  - \( x_1 \in [-22, -20, ..., 22] \) (step = 2)
  - \( x_2 \in [-10, -9, ..., 10] \) (step = 1)

### 🔵 Test Set 1 (2,500 samples)

- \( x_1 \in [-20, 20] \) (50 evenly spaced values)
- \( x_2 \in [-10, 10] \) (50 evenly spaced values)

### 🔴 Test Set 2 (2,500 samples)

- \( x_1 \in [-50, 50] \) (50 evenly spaced values)
- \( x_2 \in [-30, 30] \) (50 evenly spaced values)

---

## ⚙️ Model Architecture

- Input layer: 2 neurons (for \( x_1, x_2 \))
- Hidden layers: Configurable neurons
- Output layer: 1 neuron (linear activation)

Both versions use:
- **MSE Loss Function**
- **Adam Optimizer**
- **Linear output layer**

---

## 🧠 Experiments

### ✅ Sigmoid Activation (Hidden Layers)

| Dataset | MSE Loss |
|---------|----------|
| Training | `...` |
| Test1    | `...` |
| Test2    | `...` |

**Training Surface**

![Sigmoid Training Output](results/training_surface_sigmoid.png)

**Test1 Surface**

![Sigmoid Test1 Output](results/test1_surface_sigmoid.png)

**Test2 Surface**

![Sigmoid Test2 Output](results/test2_surface_sigmoid.png)

---

### ✅ ReLU Activation (Hidden Layers)

| Dataset | MSE Loss |
|---------|----------|
| Training | `...` |
| Test1    | `...` |
| Test2    | `...` |



## 📈 Visualizations

All outputs are plotted as 3D surfaces:

- X-axis: \( x_1 \)
- Y-axis: \( x_2 \)
- Z-axis: \( y = f(x_1, x_2) \)

Color maps and surface grids help visualize the function approximation behavior for each activation function.

---

## 🚀 How to Run

```bash
# Generate datasets
python generate_data.py
```
# Train with sigmoid
```
python model_sigmoid.py
```
# Train with ReLU
```
python model_relu.py
```
