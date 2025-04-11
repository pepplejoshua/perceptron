import numpy as np
from models.not_gate import NotGate

if __name__ == "__main__":
    not_gate = NotGate()

    # Load weights from first run
    not_gate.load_weights("weights/not_run_1/not_gate_weights.txt")

    # Test inputs and their expected outputs
    # NOT Truth Table:
    # A | Output
    # 0 |   1
    # 1 |   0
    test_inputs = np.array([0, 1])
    expected_outputs = np.array([1, 0])

    print("\nTesting NOT gate:")
    print("Input | Expected | Predicted")
    print("-" * 30)

    correct = 0
    for x, expected in zip(test_inputs, expected_outputs):
        predicted = not_gate.forward(
            np.array([x])
        )  # Need to wrap single input in array
        correct += predicted == expected
        print(f"  {x}   |    {expected}    |     {predicted}")

    accuracy = correct / len(test_inputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
