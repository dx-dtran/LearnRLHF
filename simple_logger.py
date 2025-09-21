import os
from datetime import datetime


class TrainingLogger:
    """A minimal file-based logger for training metrics."""

    def __init__(self, training_type: str, log_dir: str = "logs") -> None:
        self.training_type = training_type
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(log_dir, f"{training_type}_{timestamp}.txt")
        self._file = open(self.path, "a", encoding="utf-8")
        self._keys: list[str] | None = None

    def log(self, metrics: dict[str, object]) -> None:
        if not metrics:
            return
        if self._keys is None:
            self._keys = sorted(metrics.keys())
            header = ",".join(self._keys)
            self._file.write(header + "\n")
        values = [metrics.get(key, "") for key in self._keys]
        line = ",".join(str(value) for value in values)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __del__(self) -> None:
        self.close()
