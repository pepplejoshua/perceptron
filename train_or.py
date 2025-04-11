from models.or_gate import OrGate
from utils import get_run_directory, plot_training_results
import numpy as np
import os


def train_or_gate():
    run_dir = get_run_directory("or")
    or_gate = OrGate()
    accuracy_history, num_epochs = or_gate.train(epochs=100, learning_rate=0.1)

    # Save weights
    weights_path = os.path.join(run_dir, "or_gate_weights.txt")
    or_gate.save_weights(weights_path)

    # Plot and save results
    plots_path = os.path.join(run_dir, "training_plots.png")
    plot_training_results(or_gate, accuracy_history, num_epochs, "OR Gate", plots_path)

    return run_dir


def test_trained_or_gate(weights_path):
    or_gate = OrGate()
    or_gate.load_weights(weights_path)

    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print("\nTesting OR gate:")
    print("Input 1 | Input 2 | Output")
    print("-" * 25)
    for x in test_inputs:
        output = or_gate.forward(x)
        print(f"   {x[0]}    |    {x[1]}    |    {output}")


if __name__ == "__main__":
    run_dir = train_or_gate()
    weights_path = os.path.join(run_dir, "or_gate_weights.txt")
    test_trained_or_gate(weights_path)
