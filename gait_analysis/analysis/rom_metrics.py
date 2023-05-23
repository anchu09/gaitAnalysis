import numpy as np


def compute_rom_mean_angle(
    signal: np.ndarray,
    phases: list[tuple[int, int]],
) -> float:
    """Compute mean ROM angle across a set of gait phase intervals.

    Returns the mean of all signal samples that fall within the given phase
    intervals. After centering, positive values indicate the hip spends more
    time in flexion during that phase; negative values indicate extension.

    Args:
        signal: 1-D ROM angle array (degrees).
        phases: List of (start, end) index pairs.

    Returns:
        Mean angle across all phase samples.

    Raises:
        ValueError: If *phases* is empty.
    """
    if not phases:
        raise ValueError("Cannot compute ROM mean angle: phases list is empty.")
    values = np.concatenate([signal[s:e] for s, e in phases])
    return float(np.nanmean(values))


def compute_rom_range(signal: np.ndarray) -> float:
    """Compute peak-to-peak ROM range.

    Args:
        signal: 1-D ROM angle array (degrees).

    Returns:
        Difference between maximum and minimum angle.

    Raises:
        ValueError: If *signal* is empty.
    """
    if signal.size == 0:
        raise ValueError("Cannot compute ROM range: signal is empty.")
    return float(np.nanmax(signal) - np.nanmin(signal))
