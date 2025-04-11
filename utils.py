import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def get_run_directory(model_name: str) -> str:
    """Create and return path for current run based on model name and run count"""
    os.makedirs("weights", exist_ok=True)
    existing_runs = [
        d for d in os.listdir("weights") if d.startswith(f"{model_name}_run_")
    ]
    run_number = len(existing_runs) + 1
    run_dir = os.path.join("weights", f"{model_name}_run_{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def plot_training_results(
    model, accuracy_history: List[float], num_epochs: int, title: str, save_path: str
):
    """Plot training results and decision boundary"""
    plt.figure(figsize=(10, 4))

    # Plot accuracy over time with integer epochs
    plt.subplot(1, 2, 1)
    epochs = np.arange(1, num_epochs + 1)  # Start from 1 to match actual epochs
    plt.plot(epochs, accuracy_history)
    plt.xticks(epochs[::5])  # Show every 5th epoch
    plt.title(f"Training Accuracy Over Time ({num_epochs} epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # Visualize decision boundary
    plt.subplot(1, 2, 2)

    if model.weights.shape[0] == 1:  # 1D input (NOT gate)
        x = np.linspace(-0.5, 1.5, 100)
        outputs = [model.forward(np.array([xi])) for xi in x]
        plt.plot(x, outputs, "b-", label="Decision Boundary")
        plt.scatter([0, 1], model.y, c=["blue", "red"], s=100, label="Training Data")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.legend()
    else:  # 2D input (OR, NOR gates)
        x = np.linspace(-0.5, 1.5, 100)
        y = np.linspace(-0.5, 1.5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = model.forward(np.array([X[i, j], Y[i, j]]))
        plt.contourf(X, Y, Z, alpha=0.4)
        colors = ["red" if label == 0 else "blue" for label in model.y]
        plt.scatter(model.X[:, 0], model.X[:, 1], c=colors)
        plt.xlabel("Input 1")
        plt.ylabel("Input 2")

    plt.title(f"{title} Decision Boundary")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
