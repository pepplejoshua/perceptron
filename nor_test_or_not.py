import numpy as np
from models.nor_gate_or_not import NorGateFromOrNot

if __name__ == "__main__":
    nor_gate = NorGateFromOrNot()

    # Test Inputs and their expected outputs
    test_inputs = np.array(
        [
            [0, 0],  # 1
            [0, 1],  # 0
            [1, 0],  # 0
            [1, 1],  # 0
        ]
    )

    expected_outputs = np.array([1, 0, 0, 0])

    print("\nTesting 2-input NOR gate built with OR and NOT gates:")
    print("x1 | x2 | Expected | Predicted")
    print("-" * 30)

    correct = 0
    for x, expected in zip(test_inputs, expected_outputs):
        predicted = nor_gate.forward(x)
        correct += predicted == expected
        print(f" {x[0]} | {x[1]} |    {expected}    |     {predicted}")

    accuracy = correct / len(test_inputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
