import argparse
import json
from pathlib import Path


def plot(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError("Metrics file is empty.")
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        [row["baseten_update"] for row in rows],
        [row["baseten_train_loss"] for row in rows],
        label="Train",
    )
    validation = [row for row in rows if "baseten_validation_loss" in row]
    axes[0].plot(
        [row["baseten_update"] for row in validation],
        [row["baseten_validation_loss"] for row in validation],
        marker="o",
        label="Validation",
    )
    axes[0].set_ylabel("Token-normalized cross entropy")
    axes[0].legend()
    axes[1].plot(
        [row["baseten_update"] for row in rows],
        [row["baseten_step_seconds"] for row in rows],
        label="API training-step time",
    )
    axes[1].set_ylabel("Seconds")
    axes[1].set_xlabel("Optimizer update")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.suptitle("Baseten Loops CORD-v2 vision LoRA SFT")
    figure.tight_layout()
    destination = path.with_name("baseten_loss_curves.png")
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics")
    print(plot(parser.parse_args().metrics))
