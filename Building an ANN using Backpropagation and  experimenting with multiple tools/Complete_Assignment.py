import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import pickle
from itertools import product


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001, optimizer='adam',
                 hidden_activation='tanh', weight_cap=None, l2_lambda=0.0):
        """Initialize neural network with configurable parameters"""
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.weight_cap = weight_cap
        self.l2_lambda = l2_lambda

        # Initialize weights with proper scaling (1/sqrt(p))
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            p = layer_sizes[i]
            scale = 1.0 / np.sqrt(p)
            self.weights.append(np.random.randn(p, layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def activation_function(self, x, is_output=False):
        if is_output:
            return self.sigmoid(x)  # Output layer always uses sigmoid
        elif self.hidden_activation == 'tanh':
            return self.tanh(x)
        elif self.hidden_activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.hidden_activation == 'relu':
            return self.relu(x)

    def activation_derivative(self, x, is_output=False):
        if is_output:
            return self.sigmoid_derivative(x)
        elif self.hidden_activation == 'tanh':
            return self.tanh_derivative(x)
        elif self.hidden_activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.hidden_activation == 'relu':
            return self.relu_derivative(x)

    def forward(self, X):
        """Forward pass with weight capping applied before computation"""
        self.activations = [X]
        self.zs = []

        # Apply weight capping if specified
        if self.weight_cap is not None:
            self.weights = [np.clip(w, -self.weight_cap, self.weight_cap) for w in self.weights]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(self.activations[-1], w) + b
            self.zs.append(z)
            a = self.activation_function(z, is_output=(i == len(self.weights) - 1))
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y, output):
        """Backward pass with L2 regularization"""
        m = X.shape[0]
        delta = (output - y) / m

        dW = []
        db = []
        # Add L2 regularization term to gradient
        reg_term = self.l2_lambda * self.weights[-1]
        dW.insert(0, np.dot(self.activations[-2].T, delta) + reg_term)
        db.insert(0, np.sum(delta, axis=0, keepdims=True))

        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.zs[i - 1])
            reg_term = self.l2_lambda * self.weights[i - 1]
            dW.insert(0, np.dot(self.activations[i - 1].T, delta) + reg_term)
            db.insert(0, np.sum(delta, axis=0, keepdims=True))

        return dW, db

    def update_parameters(self, dW, db):
        """Parameter update with Adam optimizer and weight capping"""
        self.t += 1

        for i in range(len(self.weights)):
            if self.optimizer == 'adam':
                # Adam update rules with β1=0.9
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]

                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dW[i] ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db[i] ** 2)

                # Bias correction
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

                # Update parameters
                self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            else:  # SGD
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]

            # Apply weight capping
            if self.weight_cap is not None:
                self.weights[i] = np.clip(self.weights[i], -self.weight_cap, self.weight_cap)

    def compute_regularization_loss(self):
        """Calculate L2 regularization loss"""
        if self.l2_lambda == 0:
            return 0
        total = sum(np.sum(w ** 2) for w in self.weights)
        return 0.5 * self.l2_lambda * total

    def save_weights(self, filename):
        """Save weights and biases to file using pickle"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'layer_sizes': self.layer_sizes,
                'hidden_activation': self.hidden_activation
            }, f)

    def load_weights(self, filename):
        """Load weights and biases from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']
            self.layer_sizes = data['layer_sizes']
            self.hidden_activation = data['hidden_activation']

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
              verbose=True, early_stopping=True, patience=10):
        """Training with early stopping"""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        best_weights = None
        best_biases = None

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            for start in range(0, X_train.shape[0], batch_size):
                end = min(start + batch_size, X_train.shape[0])
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward and backward pass
                output = self.forward(X_batch)
                dW, db = self.backward(X_batch, y_batch, output)
                self.update_parameters(dW, db)

            # Calculate losses
            train_output = self.forward(X_train)
            train_loss = mean_squared_error(y_train, train_output) + self.compute_regularization_loss()
            train_losses.append(train_loss)

            val_output = self.forward(X_val)
            val_loss = mean_squared_error(y_val, val_output)
            val_losses.append(val_loss)

            # Early stopping logic
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                elif epoch - best_epoch >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    self.weights = best_weights
                    self.biases = best_biases
                    break

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        return train_losses, val_losses, best_epoch if early_stopping else epochs


class Normalizer:
    def __init__(self, feature_range=(-0.9, 0.9), output_range=(0.05, 0.95)):
        self.feature_range = feature_range
        self.output_range = output_range
        self.feature_params = {}
        self.output_params = {}

    def fit_features(self, X):
        self.feature_params = {}
        for i in range(X.shape[1]):
            x_min = X[:, i].min()
            x_max = X[:, i].max()
            x_range = x_max - x_min

            # Create virtual min/max with 5% buffer
            virtual_min = x_min - 0.05 * x_range
            virtual_max = x_max + 0.05 * x_range
            virtual_range = virtual_max - virtual_min

            self.feature_params[i] = {
                'min': x_min,
                'max': x_max,
                'virtual_min': virtual_min,
                'virtual_max': virtual_max,
                'virtual_range': virtual_range
            }

    def transform_features(self, X):
        X_norm = np.zeros_like(X)
        for i in range(X.shape[1]):
            params = self.feature_params[i]
            a, b = self.feature_range
            X_norm[:, i] = a + (X[:, i] - params['virtual_min']) * (b - a) / params['virtual_range']
        return X_norm

    def inverse_transform_features(self, X_norm):
        X = np.zeros_like(X_norm)
        for i in range(X_norm.shape[1]):
            params = self.feature_params[i]
            a, b = self.feature_range
            X[:, i] = params['virtual_min'] + (X_norm[:, i] - a) * params['virtual_range'] / (b - a)
        return X

    def fit_output(self, y):
        self.output_params = {}
        y_min = y.min()
        y_max = y.max()
        y_range = y_max - y_min

        # Create virtual min/max with 5% buffer
        virtual_min = y_min - 0.05 * y_range
        virtual_max = y_max + 0.05 * y_range
        virtual_range = virtual_max - virtual_min

        self.output_params = {
            'min': y_min,
            'max': y_max,
            'virtual_min': virtual_min,
            'virtual_max': virtual_max,
            'virtual_range': virtual_range
        }

    def transform_output(self, y):
        a, b = self.output_range
        return a + (y - self.output_params['virtual_min']) * (b - a) / self.output_params['virtual_range']

    def inverse_transform_output(self, y_norm):
        a, b = self.output_range
        return self.output_params['virtual_min'] + (y_norm - a) * self.output_params['virtual_range'] / (b - a)


def mean_absolute_percentage_error(y_true, y_pred, epsilon=0.001):
    """Calculate MAPE with small epsilon to avoid division by zero"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def load_data_from_excel(file_path):
    """Load data from all sheets in Excel file"""
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names

    all_X = []
    all_y = []

    for sheet in sheets:
        data = pd.read_excel(file_path, sheet_name=sheet)
        X = data[['AT', 'V', 'AP', 'RH']].values
        y = data['PE'].values.reshape(-1, 1)
        all_X.append(X)
        all_y.append(y)

    X = np.vstack(all_X)
    y = np.vstack(all_y)

    return X, y


def plot_individual_comparison(results, title_prefix, xlabel="Epoch", ylabel="MSE Loss"):
    """Plot individual training/validation curves for each configuration"""
    for key, result in results.items():
        plt.figure(figsize=(10, 6))
        plt.plot(result['train_loss'], label='Training Loss', linestyle='--')
        plt.plot(result['val_loss'], label='Validation Loss', linestyle='-')
        plt.title(f"{title_prefix} {key}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.show()


def plot_histogram_comparison(results, title, xlabel, ylabel):
    """Plot histogram comparison of test losses"""
    plt.figure(figsize=(12, 6))
    labels = [str(key) for key in results.keys()]
    values = [result['test_loss'] for result in results.values()]
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()


def run_batch_size_comparison(X_train_norm, y_train_norm, X_val_norm, y_val_norm, output_normalizer, y_test):
    """Tasks 1-4: Batch Size Comparison"""
    batch_sizes = [1, 64, 256, len(X_train_norm)]
    results = {}

    print("\n=== Tasks 1-4: Batch Size Comparison ===")

    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")
        nn = NeuralNetwork(layer_sizes=[4, 64, 32, 1],
                           learning_rate=0.001,
                           hidden_activation='tanh')

        train_loss, val_loss, _ = nn.train(X_train_norm, y_train_norm,
                                           X_val_norm, y_val_norm,
                                           epochs=100, batch_size=batch_size)

        test_output_norm = nn.forward(X_test_norm)
        test_output = output_normalizer.inverse_transform_output(test_output_norm)
        test_loss = mean_squared_error(y_test, test_output)

        results[batch_size] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'test_output': test_output
        }

    # Plot individual curves for each batch size
    plot_individual_comparison(results, "Batch Size")

    # Plot histogram of test losses
    plot_histogram_comparison(results, "Test MSE by Batch Size", "Batch Size", "Test MSE")

    print("\nBatch Size Test Results:")
    for batch_size, result in results.items():
        print(
            f"Batch size {batch_size}: Test MSE = {result['test_loss']:.4f}, RMSE = {np.sqrt(result['test_loss']):.4f}")

    return results


def run_activation_comparison(X_train_norm, y_train_norm, X_val_norm, y_val_norm, output_normalizer, y_test):
    """Task 5: Activation Function Comparison"""
    activation_functions = ['tanh', 'sigmoid', 'relu']
    results = {}

    print("\n=== Task 5: Activation Function Comparison ===")

    for activation in activation_functions:
        print(f"\nTraining with {activation} activation function")
        nn = NeuralNetwork(layer_sizes=[4, 64, 32, 1],
                           learning_rate=0.001,
                           hidden_activation=activation)

        train_loss, val_loss, _ = nn.train(X_train_norm, y_train_norm,
                                           X_val_norm, y_val_norm,
                                           epochs=100, batch_size=64)

        test_output_norm = nn.forward(X_test_norm)
        test_output = output_normalizer.inverse_transform_output(test_output_norm)
        test_loss = mean_squared_error(y_test, test_output)

        results[activation] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'test_output': test_output
        }

    # Plot individual curves for each activation
    plot_individual_comparison(results, "Activation Function")

    # Plot histogram of test losses
    plot_histogram_comparison(results, "Test MSE by Activation Function", "Activation", "Test MSE")

    print("\nActivation Function Test Results:")
    for activation, result in results.items():
        print(
            f"Activation {activation}: Test MSE = {result['test_loss']:.4f}, RMSE = {np.sqrt(result['test_loss']):.4f}")

    return results


def run_optimizer_comparison(X_train_norm, y_train_norm, X_val_norm, y_val_norm):
    """Tasks 6-7: Optimizer Comparison"""
    optimizers = ['adam', 'sgd']
    results = {}

    print("\n=== Tasks 6-7: Optimizer Comparison ===")

    for opt in optimizers:
        print(f"\nTraining with optimizer: {opt}")
        nn = NeuralNetwork(layer_sizes=[4, 64, 32, 1],
                           learning_rate=0.001,
                           optimizer=opt,
                           hidden_activation='tanh')

        train_loss, val_loss, _ = nn.train(X_train_norm, y_train_norm,
                                           X_val_norm, y_val_norm,
                                           epochs=100, batch_size=64)

        results[opt] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'final_train': train_loss[-1],
            'final_val': val_loss[-1]
        }

    # Plot comparison
    plt.figure(figsize=(12, 6))
    for opt, result in results.items():
        plt.semilogy(result['train_loss'], label=f'Train ({opt})', linestyle='--')
        plt.semilogy(result['val_loss'], label=f'Val ({opt})', linestyle='-')
    plt.title('Tasks 6-7: Optimizer Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Log MSE Loss')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    print("\nOptimizer Results:")
    print("Optimizer\tFinal Train Loss\tFinal Val Loss")
    for opt in results.keys():
        print(f"{opt}\t\t{results[opt]['final_train']:.6f}\t\t{results[opt]['final_val']:.6f}")

    return results


def run_learning_rate_experiments(X_train_norm, y_train_norm, X_val_norm, y_val_norm):
    """Task 8: Learning Rate Analysis"""
    learning_rates = [0.01, 0.001, 0.0001]
    results = {}

    print("\n=== Task 8: Learning Rate Analysis ===")

    for lr in learning_rates:
        print(f"\nTraining with η = {lr:.5f}")
        nn = NeuralNetwork(layer_sizes=[4, 64, 32, 1],
                           learning_rate=lr,
                           hidden_activation='tanh',
                           l2_lambda=0.0)

        train_loss, val_loss, _ = nn.train(X_train_norm, y_train_norm,
                                           X_val_norm, y_val_norm,
                                           epochs=100, batch_size=64)

        results[lr] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'final_train': train_loss[-1],
            'final_val': val_loss[-1]
        }

    # Plot comparison
    plt.figure(figsize=(12, 6))
    for lr, result in results.items():
        plt.semilogy(result['train_loss'], label=f'Train (η={lr:.5f})', linestyle='--')
        plt.semilogy(result['val_loss'], label=f'Val (η={lr:.5f})', linestyle='-')
    plt.title('Task 8: Learning Rate Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Log MSE Loss')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    print("\nLearning Rate Results:")
    print("η\t\tFinal Train Loss\tFinal Val Loss")
    for lr in sorted(results.keys(), reverse=True):
        print(f"{lr:.5f}\t{results[lr]['final_train']:.6f}\t{results[lr]['final_val']:.6f}")

    return results


def run_l2_regularization_experiments(X_train_norm, y_train_norm, X_val_norm, y_val_norm):
    """Task 9: L2 Regularization Analysis"""
    l2_lambdas = [0.0, 0.1, 0.5, 0.95]
    results = {}

    print("\n=== Task 9: L2 Regularization Analysis ===")

    for l2 in l2_lambdas:
        print(f"\nTraining with λ = {l2:.2f}")
        nn = NeuralNetwork(layer_sizes=[4, 64, 32, 1],
                           learning_rate=0.001,
                           hidden_activation='tanh',
                           l2_lambda=l2)

        train_loss, val_loss, _ = nn.train(X_train_norm, y_train_norm,
                                           X_val_norm, y_val_norm,
                                           epochs=100, batch_size=64)

        results[l2] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'final_train': train_loss[-1],
            'final_val': val_loss[-1]
        }

    # Plot comparison
    plt.figure(figsize=(12, 6))
    for l2, result in results.items():
        plt.semilogy(result['train_loss'], label=f'Train (λ={l2:.2f})', linestyle='--')
        plt.semilogy(result['val_loss'], label=f'Val (λ={l2:.2f})', linestyle='-')
    plt.title('Task 9: L2 Regularization Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Log MSE Loss')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    print("\nL2 Regularization Results:")
    print("λ\tFinal Train Loss\tFinal Val Loss")
    for l2 in sorted(results.keys()):
        print(f"{l2:.2f}\t{results[l2]['final_train']:.6f}\t{results[l2]['final_val']:.6f}")

    return results


def run_early_stopping_experiment(X_train_norm, y_train_norm, X_val_norm, y_val_norm,
                                  X_test_norm, y_test_norm, output_normalizer, y_test):
    """Task 10: Early Stopping and Forward-Only Mode"""
    print("\n=== Task 10: Early Stopping Experiment ===")

    # Train with early stopping
    nn = NeuralNetwork(layer_sizes=[4, 64, 32, 1],
                       learning_rate=0.001,
                       hidden_activation='tanh')

    train_loss, val_loss, best_epoch = nn.train(X_train_norm, y_train_norm,
                                                X_val_norm, y_val_norm,
                                                epochs=200, batch_size=64,
                                                early_stopping=True, patience=10)

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Early Stopping Point')
    plt.title('Task 10: Training and Validation Loss with Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"\nEarly stopping occurred at epoch {best_epoch}")

    # Save weights to file
    weights_file = 'best_weights.pkl'
    nn.save_weights(weights_file)
    print(f"Weights saved to {weights_file}")

    # Create new network and load weights
    nn2 = NeuralNetwork(layer_sizes=[4, 64, 32, 1],
                        hidden_activation='tanh')
    nn2.load_weights(weights_file)

    # Evaluate in forward-only mode
    test_output_norm = nn2.forward(X_test_norm)
    test_output = output_normalizer.inverse_transform_output(test_output_norm)

    # Calculate metrics
    test_mse = mean_squared_error(y_test, test_output)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_output)
    test_mape = mean_absolute_percentage_error(y_test, test_output)

    print("\nForward-only Mode Test Results:")
    print(f"MSE: {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"MAPE: {test_mape:.4f}%")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_output, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual PE')
    plt.ylabel('Predicted PE')
    plt.title('Task 10: Actual vs Predicted (Forward-only Mode)')
    plt.grid()
    plt.show()

    return {
        'best_epoch': best_epoch,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape
    }

def main():
    # Load data from all sheets
    file_path = "C:/Users/rosha/Documents/M.Tech/Mahindra University/SEMESTER_2/DeepLearning/Assignments/CCPP/Folds5x2_pp.xlsx"
    X, y = load_data_from_excel(file_path)

    # Data splitting (72:18:10)
    total_samples = len(X)
    test_size = int(0.10 * total_samples)
    global X_test, X_test_norm, y_test
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    X_remaining = X[:-test_size]
    y_remaining = y[:-test_size]

    # Creating a loop where index divisible by 5 is sent to "validation data" while all else is sent to "training data"
    train_indices = [i for i in range(len(X_remaining)) if i % 5 != 0]
    val_indices = [i for i in range(len(X_remaining)) if i % 5 == 0]

    X_train = X_remaining[train_indices]
    y_train = y_remaining[train_indices]
    X_val = X_remaining[val_indices]
    y_val = y_remaining[val_indices]

    # Initialize normalizers
    feature_normalizer = Normalizer(feature_range=(-0.9, 0.9))
    output_normalizer = Normalizer(output_range=(0.05, 0.95))

    # Fit and transform data
    feature_normalizer.fit_features(X_train)
    X_train_norm = feature_normalizer.transform_features(X_train)
    X_val_norm = feature_normalizer.transform_features(X_val)
    X_test_norm = feature_normalizer.transform_features(X_test)

    output_normalizer.fit_output(y_train)
    y_train_norm = output_normalizer.transform_output(y_train)
    y_val_norm = output_normalizer.transform_output(y_val)
    y_test_norm = output_normalizer.transform_output(y_test)

    # Run all experiments
    print("\nRunning all experiments...")

    # Tasks 1-4: Batch Size Comparison
    batch_results = run_batch_size_comparison(X_train_norm, y_train_norm, X_val_norm, y_val_norm, output_normalizer,
                                              y_test)

    # Task 5: Activation Function Comparison
    activation_results = run_activation_comparison(X_train_norm, y_train_norm, X_val_norm, y_val_norm,
                                                   output_normalizer, y_test)

    # Tasks 6-7: Optimizer Comparison
    optimizer_results = run_optimizer_comparison(X_train_norm, y_train_norm, X_val_norm, y_val_norm)

    # Task 8: Learning Rate Analysis
    lr_results = run_learning_rate_experiments(X_train_norm, y_train_norm, X_val_norm, y_val_norm)

    # Task 9: L2 Regularization Analysis
    l2_results = run_l2_regularization_experiments(X_train_norm, y_train_norm, X_val_norm, y_val_norm)

    # Task 10: Early Stopping and Forward-Only Mode
    early_stop_results = run_early_stopping_experiment(
        X_train_norm, y_train_norm, X_val_norm, y_val_norm,
        X_test_norm, y_test_norm, output_normalizer, y_test
    )



    # Print final summary
    print("\n=== FINAL SUMMARY OF ALL TASKS ===")

    print("\nTasks 1-4: Batch Size Comparison")
    for bs, res in batch_results.items():
        print(f"Batch {bs}: Test MSE={res['test_loss']:.4f}, RMSE={np.sqrt(res['test_loss']):.4f}")

    print("\nTask 5: Activation Function Comparison")
    for act, res in activation_results.items():
        print(f"{act}: Test MSE={res['test_loss']:.4f}, RMSE={np.sqrt(res['test_loss']):.4f}")

    print("\nTasks 6-7: Optimizer Comparison")
    for opt, res in optimizer_results.items():
        print(f"{opt}: Final Val Loss={res['final_val']:.6f}")

    print("\nTask 8: Learning Rate Analysis")
    for lr, res in lr_results.items():
        print(f"η={lr:.5f}: Final Val Loss={res['final_val']:.6f}")

    print("\nTask 9: L2 Regularization Analysis")
    for l2, res in l2_results.items():
        print(f"λ={l2:.2f}: Final Val Loss={res['final_val']:.6f}")

    print("\nTask 10: Early Stopping Results")
    print(f"Best epoch: {early_stop_results['best_epoch']}")
    print(f"Test MSE: {early_stop_results['test_mse']:.4f}")
    print(f"Test MAPE: {early_stop_results['test_mape']:.4f}%")




if __name__ == "__main__":
    main()