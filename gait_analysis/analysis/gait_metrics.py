import numpy as np

from gait_analysis.schema import STANCE


def compute_cadence(n_steps: int, duration_s: float) -> float:
    """Compute walking cadence in steps per minute.

    Args:
        n_steps: Total number of steps detected.
        duration_s: Trial duration in seconds.

    Returns:
        Cadence in steps/min.

    Raises:
        ValueError: If *duration_s* is zero or negative.
    """
    if duration_s <= 0:
        raise ValueError(f"Trial duration must be positive, got {duration_s}.")
    return 60.0 * n_steps / duration_s


def compute_phase_percentage(
    phases: list[tuple[int, int]],
    total_samples: int,
) -> float:
    """Compute the fraction of total samples occupied by a set of phases.

    Args:
        phases: List of (start, end) index pairs.
        total_samples: Total number of samples in the signal.

    Returns:
        Percentage in [0, 100].

    Raises:
        ValueError: If *total_samples* is zero.
    """
    if total_samples <= 0:
        raise ValueError(f"total_samples must be positive, got {total_samples}.")
    occupied = sum(e - s for s, e in phases)
    return 100.0 * occupied / total_samples


def compute_sls_dls(
    left_phases: dict[str, list[tuple[int, int]]],
    right_phases: dict[str, list[tuple[int, int]]],
    total_samples: int,
) -> tuple[float, float, float]:
    """Compute Single Limb Support (SLS) and Double Limb Support (DLS) percentages.

    SLS occurs when only one leg is in stance (the other is in swing).
    DLS occurs when both legs are simultaneously in stance.

    Args:
        left_phases: Phase dict for the left leg — must contain key ``'stance'``.
        right_phases: Phase dict for the right leg — must contain key ``'stance'``.
        total_samples: Total signal length in samples (must be the same for both legs).

    Returns:
        Tuple of (sls_left_pct, sls_right_pct, dls_pct), each expressed as a
        percentage of *total_samples*. The three values sum to 100 % when the
        phase intervals cover the full signal (i.e. every sample belongs to at
        least one stance phase). In practice they may sum to less than 100 %
        because samples at the trial boundaries are not assigned to any phase.

    Raises:
        ValueError: If *total_samples* is not positive, or if either phase dict
            is missing the ``'stance'`` key.
    """
    if total_samples <= 0:
        raise ValueError(f"total_samples must be positive, got {total_samples}.")
    if STANCE not in left_phases:
        raise ValueError("left_phases is missing required key 'stance'.")
    if STANCE not in right_phases:
        raise ValueError("right_phases is missing required key 'stance'.")

    def _mask(phases: list[tuple[int, int]], n: int) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        for s, e in phases:
            m[s:e] = True
        return m

    left_stance = _mask(left_phases[STANCE], total_samples)
    right_stance = _mask(right_phases[STANCE], total_samples)

    sls_left = int(np.sum(left_stance & ~right_stance))
    sls_right = int(np.sum(right_stance & ~left_stance))
    dls = int(np.sum(left_stance & right_stance))

    return (
        100.0 * sls_left / total_samples,
        100.0 * sls_right / total_samples,
        100.0 * dls / total_samples,
    )


def compute_symmetry(left_val: float, right_val: float) -> float:
    """Compute symmetry index between left and right limb metrics.

    Symmetry index = |left - right| / ((left + right) / 2) × 100.
    A value of 0 means perfect symmetry.

    Args:
        left_val: Metric value for the left limb.
        right_val: Metric value for the right limb.

    Returns:
        Symmetry index in [0, 200]. Returns NaN when both values are zero —
        this is a valid physiological edge case (both limbs at rest / no signal)
        where the symmetry index is undefined, not a programming error.
    """
    mean = (left_val + right_val) / 2.0
    if mean == 0:
        return float("nan")
    return 100.0 * abs(left_val - right_val) / mean
