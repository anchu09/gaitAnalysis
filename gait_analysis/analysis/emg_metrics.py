import numpy as np


def compute_mean_amplitude(
    signal: np.ndarray,
    phases: list[tuple[int, int]],
) -> float:
    """Compute mean EMG amplitude over a set of phase intervals.

    Args:
        signal: 1-D EMG amplitude array.
        phases: List of (start, end) index pairs defining the intervals.

    Returns:
        Mean amplitude across all phase samples.

    Raises:
        ValueError: If *phases* is empty.
    """
    if not phases:
        raise ValueError("Cannot compute mean amplitude: phases list is empty.")
    values = np.concatenate([signal[s:e] for s, e in phases])
    return float(np.nanmean(values))


def compute_mad(signal: np.ndarray) -> float:
    """Compute Muscle Activation Duration (MAD) as a percentage.

    MAD = (integral of signal during active periods) / (total integral) × 100.
    'Active' is defined as samples above the signal mean — consistent with the
    binary contraction mask used in :mod:`~gait_analysis.preprocessing.emg`.

    Args:
        signal: 1-D EMG amplitude array (non-negative, NaN-free).

    Returns:
        MAD percentage in [0, 100].

    Raises:
        ValueError: If the total signal integral is zero (all-zero signal).
    """
    total = float(np.nansum(signal))
    if total == 0:
        raise ValueError("Cannot compute MAD: signal integral is zero (all-zero or empty signal).")
    threshold = float(np.nanmean(signal))
    active_integral = float(np.nansum(signal[signal > threshold]))
    return 100.0 * active_integral / total


def compute_coactivation_index(
    agonist: np.ndarray,
    antagonist: np.ndarray,
) -> float:
    """Compute the Coactivation Index (CI) between two antagonist muscles.

    Formula (from the paper):
        CI = 2 × min(Ag, Antag) / (Ag + Antag) × 100

    where Ag and Antag are the total signal integrals (areas under the envelope).

    Args:
        agonist: 1-D EMG envelope array for the agonist muscle (e.g. tibialis).
        antagonist: 1-D EMG envelope array for the antagonist (e.g. gastrocnemius).

    Returns:
        CI in [0, 100]. Returns NaN if both integrals are zero (no muscle activity
        detected in either muscle — a valid physiological edge case where CI is
        undefined, not a programming error).

    Notes:
        A CI of 100 means both muscles are equally active at all times.
        A CI of 0 means there is no overlap in their activity.
    """
    ag = float(np.nansum(agonist))
    antag = float(np.nansum(antagonist))
    denom = ag + antag
    if denom == 0:
        return float("nan")
    return 100.0 * 2.0 * min(ag, antag) / denom
