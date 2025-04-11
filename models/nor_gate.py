from typing import List
import numpy as np
from .perceptron import SimpleOGPerceptron


class NorGate(SimpleOGPerceptron):
    def __init__(self) -> None:
        super().__init__(input_dim=2)
        # OR gate truth table
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([1, 0, 0, 0])

    def train(self, epochs: int = 100, learning_rate: float = 0.1) -> List[float]:
        """
        Train the OR gate perceptron
        Args:
            epochs: number of training iterations
            learning_rate: how much to adjust the weights by
        Returns:
            accuracy_history: list of accuracies per epoch
        """
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
