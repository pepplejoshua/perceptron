import numpy as np
from models.nand_gate import NandGate

if __name__ == "__main__":
    nand_gate = NandGate()
    nand_gate.load_weights("weights/nand_run_1/nand_gate_weights.txt")

    # Test Inputs and their expected outputs
    test_inputs = np.array(
        [
            [0, 0],  # 1
            [0, 1],  # 1
            [1, 0],  # 1
            [1, 1],  # 0
        ]
    )

    expected_outputs = np.array([1, 1, 1, 0])

    print("\nTesting 2-input NAND gate:")
    print("x1 | x2 | Expected | Predicted")
    print("-" * 30)

    correct = 0
    for x, expected in zip(test_inputs, expected_outputs):
        predicted = nand_gate.forward(x)
        correct += predicted == expected
        print(f" {x[0]} | {x[1]} |    {expected}    |     {predicted}")

    accuracy = correct / len(test_inputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
