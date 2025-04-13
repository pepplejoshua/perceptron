from models.and_gate import AndGate
from utils import get_run_directory, plot_training_results
import numpy as np
import os


def train_and_gate():
    run_dir = get_run_directory("and")
    and_gate = AndGate()
    accuracy_history, num_epochs = and_gate.train(epochs=100, learning_rate=0.1)

    # Save weights
    weights_path = os.path.join(run_dir, "and_gate_weights.txt")
    and_gate.save_weights(weights_path)

    # Plot and save results
    plots_path = os.path.join(run_dir, "training_plots.png")
    plot_training_results(
        and_gate, accuracy_history, num_epochs, "AND Gate", plots_path
    )

    return run_dir


def test_trained_and_gate(weights_path):
    and_gate = AndGate()
    and_gate.load_weights(weights_path)

    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print("\nTesting AND gate:")
    print("Input 1 | Input 2 | Output")
    print("-" * 25)
    for x in test_inputs:
        output = and_gate.forward(x)
        print(f"   {x[0]}    |    {x[1]}    |    {output}")


if __name__ == "__main__":
    run_dir = train_and_gate()
    weights_path = os.path.join(run_dir, "and_gate_weights.txt")
    test_trained_and_gate(weights_path)
