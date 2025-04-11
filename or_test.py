import numpy as np
from models.or_gate_3_inputs import ThreeInputOrGate

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
        print(f" {x[0]} | {x[1]} | {x[2]} |    {expected}    |     {predicted}")

    accuracy = correct / len(test_inputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
