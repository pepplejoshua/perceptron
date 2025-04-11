from models.or_gate import OrGate
from typing import List
import numpy as np


class ThreeInputOrGate:
    def __init__(self) -> None:
        weights_and_bias = "weights/run_1/or_gate_weights.txt"

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


if __name__ == "__main__":
    three_input_or = ThreeInputOrGate()

    # Test inputs and their expected outputs
    test_inputs = np.array(
        [
            [0, 0, 0],  # 0
            [0, 0, 1],  # 1
            [0, 1, 0],  # 1
            [0, 1, 1],  # 1
            [1, 0, 0],  # 1
            [1, 0, 1],  # 1
            [1, 1, 0],  # 1
            [1, 1, 1],  # 1
        ]
    )
    expected_outputs = np.array([0, 1, 1, 1, 1, 1, 1, 1])

    print("\nTesting 3-input OR gate:")
    print("x1 | x2 | x3 | Expected | Predicted")
    print("-" * 35)

    correct = 0
    for x, expected in zip(test_inputs, expected_outputs):
        predicted = three_input_or.forward(x)
        correct += predicted == expected
        print(f" {x[0]}  | {x[1]}  | {x[2]}  |    {expected}     |     {predicted}")

    accuracy = correct / len(test_inputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
