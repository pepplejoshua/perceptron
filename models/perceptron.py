import numpy as np


class SimpleOGPerceptron:
    def __init__(self, input_dim: int) -> None:
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn(1)

    def forward(self, x) -> int:
        """
        Forward pass of the perceptron
        Args:
            x: input vector (2D for logic gates)
        Returns:
            1 if activated, 0 if not
        """
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
