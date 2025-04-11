import os


def get_run_directory(model_name: str) -> str:
    """Create and return path for current run based on model name and run count
    Args:
        model_name: Name of the model being trained (e.g., 'or', 'not', 'nor', etc)
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
