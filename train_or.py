import numpy as np
import matplotlib.pyplot as plt
from models.or_gate import OrGate
import os


def get_run_directory(model_name: str = "or") -> str:
    """Create and return path for current run based on model name and run count
    Args:
        model_name: Name of the model being trained (e.g., 'or', 'not')
    Returns:
        Path to the new run directory
    """
    # Create weights directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)

    # Count existing runs for this model
    existing_runs = [
        d for d in os.listdir("weights") if d.startswith(f"{model_name}_run_")
    ]
    run_number = len(existing_runs) + 1

    # Create new run directory
    run_dir = os.path.join("weights", f"{model_name}_run_{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def train_or_gate():
    # Create run directory
    run_dir = get_run_directory()

    # Create and train the OR gate
    or_gate = OrGate()
    accuracy_history = or_gate.train(epochs=100, learning_rate=0.1)

    # Save the weights after training
    weights_path = os.path.join(run_dir, "or_gate_weights.txt")
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
    plt.title("OR Gate Decision Boundary")
    plt.grid(True)

    plt.tight_layout()

    # Save the plots
    plots_path = os.path.join(run_dir, "training_plots.png")
    plt.savefig(plots_path)
    plt.show()

    return run_dir


def test_trained_or_gate(weights_path):
    # Create a new OR gate and load trained weights
    or_gate = OrGate()
    or_gate.load_weights(weights_path)

    # Test all possible inputs
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print("\nTesting OR gate:")
    print("Input 1 | Input 2 | Output")
    print("-" * 25)
    for x in test_inputs:
        output = or_gate.forward(x)
        print(f"   {x[0]}    |    {x[1]}    |    {output}")


if __name__ == "__main__":
    # Train and save the model
    run_dir = train_or_gate()

    # Test the trained model using the weights from this run
    weights_path = os.path.join(run_dir, "or_gate_weights.txt")
    test_trained_or_gate(weights_path)
