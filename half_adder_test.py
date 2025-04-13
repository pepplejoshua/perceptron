import numpy as np
from models.half_adder import HalfAdder

if __name__ == "__main__":
    half_adder = HalfAdder()

    # Test Inputs and their expected outputs
    test_inputs = np.array(
        [  #           XOR | AND
            [0, 0],  #  0  |  0
            [0, 1],  #  1  |  0
            [1, 0],  #  1  |  0
            [1, 1],  #  0  |  1
        ]
    )

    expected_outputs = np.array([[0, 0], [1, 0], [1, 0], [0, 1]])

    print("\nTesting 2-input Half-Adder:")
    print("x1 | x2 | Expected (Sum, Carry) | Predicted (Sum, Carry)")
    print("-" * 56)

    correct = 0
    for x, (sum_i, carry_i) in zip(test_inputs, expected_outputs):
        pred_sum, pred_carry = half_adder.forward(x)
        correct += pred_sum == sum_i and pred_carry == carry_i
        print(
            f" {x[0]} | {x[1]}  |         ({sum_i}, {carry_i})        |       ({pred_sum}, {pred_carry})"
        )

    accuracy = correct / len(test_inputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
