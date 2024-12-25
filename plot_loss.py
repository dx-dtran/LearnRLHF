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

    timestamps = []
    losses = []

    for line in log_lines:
        match = re.search(r"Batch \[\d+/\d+\], Loss: ([\d\.]+)", line)
        if match:
            loss = float(match.group(1))
            losses.append(loss)
            timestamps.append(len(timestamps) + 1)

    smoothed_losses = smooth(losses, alpha=smoothing_factor)
    return timestamps, losses, smoothed_losses


def load_and_plot_loss(log_file_path, smoothing_factor=0.1):
    timestamps, losses, smoothed_losses = load_loss_from_file(
        log_file_path, smoothing_factor
    )

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, losses, label="Original Loss", alpha=0.5)
    plt.plot(
        timestamps,
        smoothed_losses,
        label=f"Smoothed Loss (α={smoothing_factor})",
        linewidth=2,
    )
    plt.xlabel("Time (log index)")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()
    plt.show()


def load_and_plot_validation_loss(log_file_path):
    with open(log_file_path, "r") as file:
        log_lines = file.readlines()

    batch_losses = []
    batch_loss_timestamps = []
    validation_losses = []
    validation_timestamps = []

    for i, line in enumerate(log_lines):
        batch_match = re.search(r"Batch \[\d+/\d+\], Loss: ([\d\.]+)", line)
        if batch_match:
            batch_losses.append(float(batch_match.group(1)))
            batch_loss_timestamps.append(len(batch_losses))

        validation_match = re.search(r"Validation Subset Loss: ([\d\.]+)", line)
        if validation_match:
            validation_losses.append(float(validation_match.group(1)))
            validation_timestamps.append(len(batch_losses))

    plt.figure(figsize=(12, 6))
    plt.plot(
        batch_loss_timestamps,
        batch_losses,
        label="Training Batch Loss (mini-batches)",
        alpha=0.6,
        linestyle="--",
    )
    plt.scatter(
        validation_timestamps,
        validation_losses,
        color="red",
        label="Validation Loss",
        zorder=3,
    )
    plt.xlabel("Iteration Index")
    plt.ylabel("Loss")
    plt.title("Validation Loss Over Time")
    plt.legend()
    plt.grid()
    plt.show()


def compare_multiple_losses(log_files, smoothing_factor=0.1, max_iterations=None):
    plt.figure(figsize=(12, 6))

    for i, log_file in enumerate(log_files):
        timestamps, losses, smoothed_losses = load_loss_from_file(
            log_file, smoothing_factor
        )

        if max_iterations is not None:
            timestamps, losses, smoothed_losses = [
                x[:max_iterations] for x in (timestamps, losses, smoothed_losses)
            ]

        plt.plot(
            timestamps,
            smoothed_losses,
            label=f"Smoothed Loss (File {i+1}, α={smoothing_factor})",
            linewidth=2,
        )
        plt.plot(
            timestamps,
            losses,
            label=f"Original Loss (File {i+1})",
            alpha=0.5,
        )

    plt.xlabel("Time (log index)")
    plt.ylabel("Loss")
    plt.title(
        f"Comparison of Training Losses (up to iteration {max_iterations if max_iterations else 'all'})"
    )
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    smoothing_factor = 0.1
    log_file_path = "training-12-25-2024-21-04-14.log"
    load_and_plot_loss(log_file_path, smoothing_factor=smoothing_factor)
    load_and_plot_validation_loss(log_file_path)
