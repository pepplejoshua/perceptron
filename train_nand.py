from models.nand_gate import NandGate
from utils import get_run_directory, plot_training_results
import numpy as np
import os


def train_nand_gate():
    run_dir = get_run_directory("nand")
    nand_gate = NandGate()
    accuracy_history, num_epochs = nand_gate.train(epochs=100, learning_rate=0.1)

    # Save weights
    weights_path = os.path.join(run_dir, "nand_gate_weights.txt")
    nand_gate.save_weights(weights_path)

    # Plot and save results
    plots_path = os.path.join(run_dir, "training_plots.png")
    plot_training_results(
        nand_gate, accuracy_history, num_epochs, "NAND Gate", plots_path
    )

    return run_dir


def test_trained_nand_gate(weights_path):
    nand_gate = NandGate()
    nand_gate.load_weights(weights_path)

    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print("\nTesting NAND gate:")
    print("Input 1 | Input 2 | Output")
    print("-" * 25)
    for x in test_inputs:
        output = nand_gate.forward(x)
        print(f"   {x[0]}    |    {x[1]}    |    {output}")


if __name__ == "__main__":
    run_dir = train_nand_gate()
    weights_path = os.path.join(run_dir, "nand_gate_weights.txt")
    test_trained_nand_gate(weights_path)
