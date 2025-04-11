from models.nor_gate import NorGate
from utils import get_run_directory, plot_training_results
import os
import numpy as np


def train_nor_gate():
    run_dir = get_run_directory("nor")
    nor_gate = NorGate()
    accuracy_history, num_epochs = nor_gate.train(epochs=100, learning_rate=0.1)

    # Save weights
    weights_path = os.path.join(run_dir, "nor_gate_weights.txt")
    nor_gate.save_weights(weights_path)

    # Plot and save results
    plots_path = os.path.join(run_dir, "training_plots.png")
    plot_training_results(
        nor_gate, accuracy_history, num_epochs, "NOR Gate", plots_path
    )

    return run_dir


def test_trained_nor_gate(weights_path):
    nor_gate = NorGate()
    nor_gate.load_weights(weights_path)

    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print("\nTesting NOR gate:")
    print("Input 1 | Input 2 | Output")
    print("-" * 25)
    for x in test_inputs:
        output = nor_gate.forward(x)
        print(f"   {x[0]}    |    {x[1]}    |    {output}")


if __name__ == "__main__":
    run_dir = train_nor_gate()
    weights_path = os.path.join(run_dir, "nor_gate_weights.txt")
    test_trained_nor_gate(weights_path)
