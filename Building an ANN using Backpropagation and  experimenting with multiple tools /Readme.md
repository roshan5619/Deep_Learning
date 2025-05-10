# ANN with Backpropagation

## Overview

This project involves building an Artificial Neural Network (ANN) from scratch using backpropagation. The assignment emphasizes understanding the nuts and bolts of backpropagation, vectorization in Python, and experimenting with various optimization techniques—all without using high-level deep learning libraries such as TensorFlow, Keras, or PyTorch.

The ANN is first trained on a toy problem of approximating the function:
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**y = sin(x), where -2π ≤ x ≤ 2π**

After validating the network on this toy dataset, the ANN is further tested on the Combined Cycle Power Plant dataset (available from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)) to demonstrate its capability on real-world regression tasks.

## Table of Contents

- [Project Objectives](#project-objectives)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Implementation Details](#implementation-details)
  - [Data Splitting](#data-splitting)
  - [Network Architecture](#network-architecture)
  - [Backpropagation and Vectorization](#backpropagation-and-vectorization)
  - [Mini-Batch Granulation](#mini-batch-granulation)
  - [Activation Functions](#activation-functions)
  - [I/O Normalization](#io-normalization)
  - [Weight Initialization](#weight-initialization)
  - [Learning Rate & Regularization](#learning-rate--regularization)
  - [Momentum & Optimization](#momentum--optimization)
  - [Stopping Criteria & Error Calculation](#stopping-criteria--error-calculation)
- [Usage Instructions](#usage-instructions)
- [Submission Guidelines](#submission-guidelines)
- [License](#license)

## Project Objectives

- **Understand and implement backpropagation:** Write out the complete algorithm with vectorized Python code.
- **Experiment with hyper-parameters:** Adjust mini-batch sizes, activation functions (tanh, logistic, and ReLU), and learning rates.
- **Error Analysis:** Evaluate the ANN using Mean Absolute Percentage Error (MAPE) for both validation and test datasets.
- **Real World Application:** Demonstrate the network’s performance on the Combined Cycle Power Plant dataset for regression.

## Requirements

- **Programming Language:** Python (preferably 3.x)
- **Libraries:**  
  - Numpy  
  - Pandas  
  - Matplotlib  
  - (No high-level deep learning libraries are allowed.)
- **Hardware:**  
  - Recommended to run the code on a GPU-capable machine such as the DGX-1 for performance testing.

## Datasets

1. **Toy Dataset:**  
   - **Function:** y = sin(x)  
   - **Domain:** x ∈ [-2π, 2π]  
   - **Training Data:** 1000 equally spaced (x, y) pairs  
   - **Validation Data:** 300 randomly generated x-values (with corresponding network outputs for comparison)
   
2. **Real Dataset:**  
   - **Dataset:** Combined Cycle Power Plant Data (more than 9,000 rows with 4 inputs and 1 output)  
   - **Source:** [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)

## Implementation Details

### Data Splitting

- **Strategy:** Split data into training, validation, and testing sets (ideally 72% : 18% : 10%).  
- **Note:** Test data should be taken as continuous chunks from one end of the dataset, while training/validation can be interleaved using a loop (e.g., every 5th sample for validation).

### Network Architecture

- **Input Layer:**  
  - 1 neuron for the toy problem (and 4 neurons for the power plant dataset).
- **Hidden Layers:**  
  - At least one hidden layer is required.  
  - The number of neurons should follow the rule of thumb (total unknowns should be fewer than half the number of training samples).
- **Output Layer:**  
  - 1 neuron for output.

### Backpropagation and Vectorization

- **Objective:** Derive and implement the backpropagation equations (as outlined in the provided slides/powerpoint).
- **Focus:**  
  - Use complete or partial vectorization to achieve a two-order-of-magnitude reduction in computation time.
  - List out the equations and corresponding control actions for clarity.

### Mini-Batch Granulation

- **Mini-Batch Sizes:** Explore sizes ranging from 1 (effectively SGD) to 64, 256, and full-batch.
- **Shuffling:** Randomize the order of mini-batches across epochs to improve convergence.

### Activation Functions

- **Hidden Layers:** You may experiment among tanh, logistic, or ReLU functions.  
- **Output Layer:** For consistency and proper scaling, use logistic or tanh (never ReLU).

### I/O Normalization

- **Normalization:**  
  - Scale inputs and outputs to the range [-1, +1] (or [0, 1] for logistic outputs).
  - Consider using a mapping that shifts actual minimum and maximum values to about -0.9 and +0.9, respectively, by introducing virtual minima and maxima.

### Weight Initialization

- **Technique:** Follow guidelines to cap the absolute weight values (e.g., at +1) to avoid large initializations and as a form of crude regularization.
- **Reference:** Slides 33 and 34 provide details on initialization strategies.

### Learning Rate & Regularization

- **Learning Rate:** Start with η = 0.001 and experiment by freezing other parameters.
- **L2 Regularization:**  
  - Investigate the impact of increasing regularization coefficients (γ) on convergence.
- **Tip:** Compare different learning rate values (e.g., 0.01, 0.0001) while keeping all else constant.

### Momentum & Optimization

- **Momentum:**  
  - Use a momentum term (β = 0.9) as per the provided equation (K).
- **Additional Optimization:**  
  - Implement Adam optimization as an optional bonus to compare against SGD-with-momentum.

### Stopping Criteria & Error Calculation

- **Early Stopping:** Monitor both training and validation errors to stop at the least overfitting point.
- **Final Evaluation:**  
  - After concluding training, run a test phase using only forward passes.
  - Calculate the Mean Absolute Percentage Error (MAPE) over the validation and testing sets.

## Usage Instructions

1. **Configuration:**  
   - All hyper-parameters (e.g., learning rate, mini-batch size, activation function type) are to be configurable via a configuration file or command-line parameters. This avoids the need to alter code for each experiment.

2. **Execution Modes:**  
   - **Training Mode:** Performs forward and backward passes; saves the final trained weights in a structured file.
   - **Test Mode:** Loads saved weights and performs forward pass calculations to compute the error on test data.

3. **Running the Code:**  
   - Run the main script using Python. For instance:  
     ```bash
     python main.py --mode train --config config.json
     ```
   - To test the network:
     ```bash
     python main.py --mode test --config config.json
     ```

4. **GPU Utilization:**  
   - When available, ensure the code is executed on a GPU (e.g., on the DGX-1 machine) to observe performance improvements.

## Submission Guidelines

- **Code Submission:** Submit a folder containing:
  - Your complete Python code.
  - A configuration file detailing the chosen hyper-parameters and experimental settings.
- **Documentation:** Provide a PDF document describing:
  - The complete step-by-step backpropagation algorithm.
  - Detailed comparisons of the impact of varying parameters (items 2, 4, 5, 8, and 10).
  - The best parameter combination(s) for each dataset along with the corresponding MAPE values.
- **Collaboration:** Ensure that the work is your own team’s effort and that no external group support was used.

## License

This project is provided for academic purposes. Redistribution or reuse outside of educational research requires proper attribution.

---

*Happy Coding and Experimenting with ANN Backpropagation!*

