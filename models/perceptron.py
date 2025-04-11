import numpy as np
from typing import List


class SimpleOGPerceptron:
    def __init__(self, input_dim: int) -> None:
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn(1)
        # Truth table to be defined by subclasses
        self.X = None
        self.y = None

    def train(self, epochs: int = 100, learning_rate: float = 0.1) -> List[float]:
        """
        Train the perceptron using its truth table
        Args:
            epochs: number of training iterations
            learning_rate: how much to adjust the weights by
        Returns:
            accuracy_history: list of accuracies per epoch
        """
        if self.X is None or self.y is None:
            raise ValueError("Truth table (X, y) must be defined before training")

        accuracy_history = []

        for epoch in range(epochs):
            correct = 0
            for x_i, y_i in zip(self.X, self.y):
                # Make a prediction
                pred = self.forward(x_i)

                # Update weights if prediction is wrong
                if pred != y_i:
                    error = y_i - pred
                    self.weights += learning_rate * error * x_i
                    self.bias += learning_rate * error
                else:
                    correct += 1

            accuracy = correct / len(self.X)
            accuracy_history.append(accuracy)

            # Early stopping if we achieve 100% accuracy
            if accuracy == 1.0:
                break

        return accuracy_history

    def forward(self, x) -> int:
        """Forward pass of the perceptron"""
        return 1 if np.dot(x, self.weights) + self.bias > 0 else 0

    def save_weights(self, filepath):
        """Save weights and bias to a simple text file"""
        weights_and_bias = np.append(self.weights, self.bias)
        np.savetxt(f"{filepath}", weights_and_bias)
        print(f"saved to {filepath}'...")

    def load_weights(self, filepath):
        """Load weights and bias from a text file"""
        weights_and_bias = np.loadtxt(f"{filepath}")
        print(f"read {filepath}'...")
        self.weights = weights_and_bias[:-1]
        self.bias = weights_and_bias[-1]
