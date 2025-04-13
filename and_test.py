import numpy as np
from models.and_gate import AndGate

if __name__ == "__main__":
    and_gate = AndGate()
    and_gate.load_weights("weights/and_run_1/and_gate_weights.txt")

    # Test Inputs and their expected outputs
    test_inputs = np.array(
        [
            [0, 0],  # 0
            [0, 1],  # 0
            [1, 0],  # 0
            [1, 1],  # 1
        ]
    )

    expected_outputs = np.array([0, 0, 0, 1])

    print("\nTesting 2-input AND gate:")
    print("x1 | x2 | Expected | Predicted")
    print("-" * 30)

    correct = 0
    for x, expected in zip(test_inputs, expected_outputs):
        predicted = and_gate.forward(x)
        correct += predicted == expected
        print(f" {x[0]} | {x[1]} |    {expected}    |     {predicted}")

    accuracy = correct / len(test_inputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
