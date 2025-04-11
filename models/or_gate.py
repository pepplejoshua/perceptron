import numpy as np
from .perceptron import SimpleOGPerceptron


class OrGate(SimpleOGPerceptron):
    def __init__(self) -> None:
        super().__init__(input_dim=2)
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([0, 1, 1, 1])
