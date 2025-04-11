from .or_gate import OrGate
from typing import List


class ThreeInputOrGate:
    def __init__(self) -> None:
        weights_and_bias = "weights/or_run_1/or_gate_weights.txt"

        # Create two OR gates with same weights
        gate_1 = OrGate()
        gate_1.load_weights(weights_and_bias)
        gate_2 = OrGate()
        gate_2.weights = gate_1.weights
        gate_2.bias = gate_1.bias

        self.gate1 = gate_1  # Will process inputs 1 and 2
        self.gate2 = gate_2  # Will process result of gate1 with input 3

    def forward(self, x: List[int]) -> int:
        """
        Process 3 inputs using two 2-input OR gates in cascade
        Args:
            x: List of 3 binary inputs [x1, x2, x3]
        Returns:
            Binary output (0 or 1)
        """
        # First OR gate processes x1 OR x2
        intermediate = self.gate1.forward(x[:2])

        # Second OR gate processes (x1 OR x2) OR x3
        final = self.gate2.forward([intermediate, x[2]])

        return final
