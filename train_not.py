import numpy as np
import matplotlib.pyplot as plt
from models.not_gate import NotGate
from utils import get_run_directory
import os


def train_not_gate():
    # Create run directory
    run_dir = get_run_directory("not")

    # Create and train the NOT gate
    not_gate = NotGate()
    accuracy_history = not_gate.train(epochs=100, learning_rate=0.1)

    # Save the weights after training
    weights_path = os.path.join(run_dir, "not_gate_weights.txt")
    not_gate.save_weights(weights_path)

    # Create figure for plots
    plt.figure(figsize=(10, 4))

    # Plot accuracy over time
    plt.subplot(1, 2, 1)
    plt.plot(accuracy_history)
    plt.title("Training Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # Visualize decision boundary
    plt.subplot(1, 2, 2)

    # Create input points for visualization
    x = np.linspace(-0.5, 1.5, 100)

    # Calculate outputs for each point
    outputs = [not_gate.forward(np.array([xi])) for xi in x]

    # Plot decision boundary
    plt.plot(x, outputs, "b-", label="Decision Boundary")

    # Plot training points
    plt.scatter([0, 1], [1, 0], c=["blue", "red"], s=100, label="Training Data")

    # Add labels and title
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("NOT Gate Decision Boundary")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save the plots
    plots_path = os.path.join(run_dir, "training_plots.png")
    plt.savefig(plots_path)
    plt.show()

    return run_dir


def test_trained_not_gate(weights_path):
    # Create a new NOT gate and load trained weights
    not_gate = NotGate()
    not_gate.load_weights(weights_path)

    # Test all possible inputs
    test_inputs = np.array([0, 1])
    print("\nTesting NOT gate:")
    print("Input | Expected | Output")
    print("-" * 25)
    for x in test_inputs:
        output = not_gate.forward(np.array([x]))
        expected = 1 if x == 0 else 0
        print(f"  {x}   |    {expected}    |   {output}")


if __name__ == "__main__":
    # Train and save the model
    run_dir = train_not_gate()

    # Test the trained model using the weights from this run
    weights_path = os.path.join(run_dir, "not_gate_weights.txt")
    test_trained_not_gate(weights_path)
