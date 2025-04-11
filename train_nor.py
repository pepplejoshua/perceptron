import numpy as np
import matplotlib.pyplot as plt
from models.nor_gate import NorGate
from utils import get_run_directory
import os


def train_nor_gate():
    # Create run directory
    run_dir = get_run_directory("nor")

    # Create and train the OR gate
    or_gate = NorGate()
    accuracy_history = or_gate.train(epochs=100, learning_rate=0.1)

    # Save the weights after training
    weights_path = os.path.join(run_dir, "nor_gate_weights.txt")
    or_gate.save_weights(weights_path)

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

    # Create a grid of points
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate decision boundary
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = or_gate.forward(np.array([X[i, j], Y[i, j]]))

    # Plot decision boundary
    plt.contourf(X, Y, Z, alpha=0.4)

    # Plot training points
    colors = ["red" if label == 0 else "blue" for label in or_gate.y]
    plt.scatter(or_gate.X[:, 0], or_gate.X[:, 1], c=colors)

    # Add labels and title
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.title("NOR Gate Decision Boundary")
    plt.grid(True)

    plt.tight_layout()

    # Save the plots
    plots_path = os.path.join(run_dir, "training_plots.png")
    plt.savefig(plots_path)
    plt.show()

    return run_dir


def test_trained_nor_gate(weights_path):
    # Create a new OR gate and load trained weights
    or_gate = NorGate()
    or_gate.load_weights(weights_path)

    # Test all possible inputs
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print("\nTesting NOR gate:")
    print("Input 1 | Input 2 | Output")
    print("-" * 25)
    for x in test_inputs:
        output = or_gate.forward(x)
        print(f"   {x[0]}    |    {x[1]}    |    {output}")


if __name__ == "__main__":
    # Train and save the model
    run_dir = train_nor_gate()

    # Test the trained model using the weights from this run
    weights_path = os.path.join(run_dir, "nor_gate_weights.txt")
    test_trained_nor_gate(weights_path)
