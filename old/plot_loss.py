import re
import matplotlib.pyplot as plt


def smooth(data, alpha=0.1):
    smoothed = []
    for i, point in enumerate(data):
        if i == 0:
            smoothed.append(point)
        else:
            smoothed_value = alpha * point + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_value)
    return smoothed


def load_loss_from_file(log_file_path, smoothing_factor=0.1):
    with open(log_file_path, "r") as file:
        log_lines = file.readlines()

    global_steps = []
    losses = []

    for line in log_lines:
        match = re.search(r"Global Step \[(\d+)\], Avg Loss: ([\d\.]+)", line)
        if match:
            global_step = int(match.group(1))
            loss = float(match.group(2))
            global_steps.append(global_step)
            losses.append(loss)

    smoothed_losses = smooth(losses, alpha=smoothing_factor)
    return global_steps, losses, smoothed_losses


def load_and_plot_loss(log_file_path, smoothing_factor=0.1):
    global_steps, losses, smoothed_losses = load_loss_from_file(
        log_file_path, smoothing_factor
    )

    plt.figure(figsize=(12, 6))
    plt.plot(global_steps, losses, label="Original Loss", alpha=0.5)
    plt.plot(
        global_steps,
        smoothed_losses,
        label=f"Smoothed Loss (α={smoothing_factor})",
        linewidth=2,
    )
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()
    plt.show()


def load_and_plot_validation_loss(log_file_path):
    with open(log_file_path, "r") as file:
        log_lines = file.readlines()

    batch_losses = []
    batch_loss_steps = []
    validation_losses = []
    validation_steps = []

    for line in log_lines:
        batch_match = re.search(r"Global Step \[(\d+)\], Avg Loss: ([\d\.]+)", line)
        if batch_match:
            global_step = int(batch_match.group(1))
            loss = float(batch_match.group(2))
            batch_losses.append(loss)
            batch_loss_steps.append(global_step)

        validation_match = re.search(r"Validation Subset Loss: ([\d\.]+)", line)
        if validation_match:
            validation_loss = float(validation_match.group(1))
            validation_losses.append(validation_loss)
            validation_steps.append(global_step)

    plt.figure(figsize=(12, 6))
    plt.plot(
        batch_loss_steps,
        batch_losses,
        label="Training Batch Loss",
        alpha=0.6,
        linestyle="--",
    )
    plt.scatter(
        validation_steps,
        validation_losses,
        color="red",
        label="Validation Loss",
        zorder=3,
    )
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Validation Loss Over Time")
    plt.legend()
    plt.grid()
    plt.show()


def compare_multiple_losses(log_files, smoothing_factor=0.1, max_steps=None):
    plt.figure(figsize=(12, 6))

    for i, log_file in enumerate(log_files):
        global_steps, losses, smoothed_losses = load_loss_from_file(
            log_file, smoothing_factor
        )

        if max_steps is not None:
            global_steps, losses, smoothed_losses = [
                x[:max_steps] for x in (global_steps, losses, smoothed_losses)
            ]

        plt.plot(
            global_steps,
            smoothed_losses,
            label=f"Smoothed Loss (File {i+1}, α={smoothing_factor})",
            linewidth=2,
        )
        plt.plot(
            global_steps,
            losses,
            label=f"Original Loss (File {i+1})",
            alpha=0.5,
        )

    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title(
        f"Comparison of Training Losses (up to step {max_steps if max_steps else 'all'})"
    )
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    smoothing_factor = 0.1
    log_file_path = "training-12-25-2024-23-26-32.log"
    load_and_plot_loss(log_file_path, smoothing_factor=smoothing_factor)
    load_and_plot_validation_loss(log_file_path)
