import logging

import numpy as np
from scipy.signal import find_peaks

from gait_analysis.config import PEAK_MIN_DISTANCE
from gait_analysis.schema import PEAK_TYPE_MAX, PEAK_TYPE_MIN, STANCE, SWING

logger = logging.getLogger(__name__)

# Structured-array dtype for the peak sequence
_PEAK_DTYPE = np.dtype([("index", int), ("type", int)])


def detect_peaks(
    signal: np.ndarray,
    min_distance: int = PEAK_MIN_DISTANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect maxima and minima in a ROM signal.

    Args:
        signal: 1-D preprocessed ROM signal.
        min_distance: Minimum number of samples between peaks.

    Returns:
        Tuple of (maxima_indices, minima_indices).

    Raises:
        ValueError: If *signal* is empty.
    """
    if signal.size == 0:
        raise ValueError("Cannot detect peaks: signal is empty.")
    height = float(np.mean(signal))
    maxima, _ = find_peaks(signal, height=height, distance=min_distance)
    minima, _ = find_peaks(-signal, height=-height, distance=min_distance)
    return maxima, minima


def build_peak_sequence(
    maxima: np.ndarray,
    minima: np.ndarray,
) -> np.ndarray:
    """Merge maxima and minima into a single chronologically sorted sequence.

    Each element carries its sample index and a type tag:
    ``PEAK_TYPE_MAX`` (+1) for maxima, ``PEAK_TYPE_MIN`` (-1) for minima.

    Args:
        maxima: Integer indices of maxima in the original signal.
        minima: Integer indices of minima in the original signal.

    Returns:
        Structured array with fields ``'index'`` and ``'type'``, sorted by index.
        Returns an empty array if both inputs are empty.
    """
    max_entries = np.array([(i, PEAK_TYPE_MAX) for i in maxima], dtype=_PEAK_DTYPE)
    min_entries = np.array([(i, PEAK_TYPE_MIN) for i in minima], dtype=_PEAK_DTYPE)
    if max_entries.size == 0 and min_entries.size == 0:
        return np.array([], dtype=_PEAK_DTYPE)
    all_peaks = np.concatenate([max_entries, min_entries])
    return np.sort(all_peaks, order="index")


def correct_peak_artifacts(
    signal: np.ndarray,
    sequence: np.ndarray,
) -> np.ndarray:
    """Remove false peaks and consecutive same-type duplicates.

    When two consecutive peaks have the same type, only the more extreme one
    is kept (largest value for maxima, smallest for minima).

    Args:
        signal: 1-D ROM signal.
        sequence: Structured peak sequence from :func:`build_peak_sequence`.

    Returns:
        Cleaned peak sequence with strictly alternating max/min types.
    """
    if sequence.size < 2:
        return sequence

    cleaned = [sequence[0]]
    for peak in sequence[1:]:
        last = cleaned[-1]
        if peak["type"] == last["type"]:
            if peak["type"] == PEAK_TYPE_MAX:
                if signal[peak["index"]] > signal[last["index"]]:
                    cleaned[-1] = peak
            else:  # PEAK_TYPE_MIN
                if signal[peak["index"]] < signal[last["index"]]:
                    cleaned[-1] = peak
        else:
            cleaned.append(peak)

    return np.array(cleaned, dtype=_PEAK_DTYPE)


def extract_phases(
    signal: np.ndarray,
    sequence: np.ndarray,
) -> dict[str, list[tuple[int, int]]]:
    """Segment the signal into swing and stance phases.

    Each consecutive max→min pair is a stance phase (hip flexing toward
    extension). Each consecutive min→max pair is a swing phase (hip extending
    toward flexion).

    Args:
        signal: 1-D ROM signal (used only for length validation).
        sequence: Cleaned peak sequence from :func:`correct_peak_artifacts`.

    Returns:
        Dict with keys ``'swing'`` and ``'stance'``, each a list of
        ``(start_idx, end_idx)`` tuples.

    Raises:
        ValueError: If the sequence has fewer than 2 peaks, or if no swing
            or no stance phases could be extracted (which indicates the ROM
            signal is too short or too noisy for analysis).
    """
    if sequence.size < 2:
        raise ValueError(
            f"Peak sequence too short ({sequence.size} peaks); need ≥ 2 to extract phases."
        )

    phases: dict[str, list[tuple[int, int]]] = {SWING: [], STANCE: []}

    for i in range(len(sequence) - 1):
        start_idx = int(sequence[i]["index"])
        end_idx = int(sequence[i + 1]["index"])
        start_type = int(sequence[i]["type"])
        end_type = int(sequence[i + 1]["type"])

        if start_type == PEAK_TYPE_MIN and end_type == PEAK_TYPE_MAX:
            phases[SWING].append((start_idx, end_idx))
        elif start_type == PEAK_TYPE_MAX and end_type == PEAK_TYPE_MIN:
            phases[STANCE].append((start_idx, end_idx))

    n_swing = len(phases[SWING])
    n_stance = len(phases[STANCE])
    if n_swing == 0 or n_stance == 0:
        raise ValueError(
            f"Phase extraction yielded {n_swing} swing and {n_stance} stance phases. "
            "Signal may be too short or noisy."
        )

    return phases
