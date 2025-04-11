import numpy as np
from .perceptron import SimpleOGPerceptron


class NotGate(SimpleOGPerceptron):
    def __init__(self) -> None:
        super().__init__(input_dim=1)
        self.X = np.array([0, 1])
        self.y = np.array([1, 0])
