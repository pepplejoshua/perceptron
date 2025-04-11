from models.not_gate import NotGate
from utils import get_run_directory, plot_training_results
import os
import numpy as np


def train_not_gate():
    run_dir = get_run_directory("not")
    not_gate = NotGate()
    accuracy_history, num_epochs = not_gate.train(epochs=100, learning_rate=0.1)

    # Save weights
    weights_path = os.path.join(run_dir, "not_gate_weights.txt")
    not_gate.save_weights(weights_path)

    # Plot and save results
    plots_path = os.path.join(run_dir, "training_plots.png")
    plot_training_results(
        not_gate, accuracy_history, num_epochs, "NOT Gate", plots_path
    )

    return run_dir


def test_trained_not_gate(weights_path):
    not_gate = NotGate()
    not_gate.load_weights(weights_path)

    test_inputs = np.array([0, 1])
    print("\nTesting NOT gate:")
    print("Input | Expected | Output")
    print("-" * 25)
    for x in test_inputs:
        output = not_gate.forward(np.array([x]))
        expected = 1 if x == 0 else 0
        print(f"  {x}   |    {expected}    |   {output}")


if __name__ == "__main__":
    run_dir = train_not_gate()
    weights_path = os.path.join(run_dir, "not_gate_weights.txt")
    test_trained_not_gate(weights_path)
