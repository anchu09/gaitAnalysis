import numpy as np
from scipy.signal import butter, filtfilt, resample

from gait_analysis.config import (
    FILTER_CUTOFF,
    FILTER_ORDER,
    INTERPOLATION_FACTOR,
    TRANSIENT_TRIM_FRACTION,
)


def interpolate_rom(signal: np.ndarray, upsample_factor: int = INTERPOLATION_FACTOR) -> np.ndarray:
    """Upsample a ROM signal using Fourier-based resampling.

    Args:
        signal: 1-D ROM angle array (degrees).
        upsample_factor: Multiplicative upsampling factor (default 10×).

    Returns:
        Upsampled array of length ``len(signal) * upsample_factor``.

    Raises:
        ValueError: If *signal* is empty.
    """
    if signal.size == 0:
        raise ValueError("Cannot interpolate ROM: signal is empty.")
    return resample(signal, len(signal) * upsample_factor)


def highpass_filter(
    signal: np.ndarray,
    cutoff: float = FILTER_CUTOFF,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter to a ROM signal.

    Uses ``scipy.signal.filtfilt`` for zero phase distortion.

    Args:
        signal: 1-D ROM angle array.
        cutoff: Normalized cutoff frequency — must be in the open interval (0, 1),
            where 1 corresponds to the Nyquist frequency.
        order: Filter order — must be ≥ 1.

    Returns:
        Filtered signal of the same length.

    Raises:
        ValueError: If *cutoff* is not in (0, 1) or *order* is < 1.
    """
    if not (0 < cutoff < 1):
        raise ValueError(f"Filter cutoff must be in (0, 1), got {cutoff}.")
    if order < 1:
        raise ValueError(f"Filter order must be ≥ 1, got {order}.")
    b, a = butter(order, cutoff, btype="high", analog=False)
    return filtfilt(b, a, signal)


def trim_transient(
    signal: np.ndarray,
    fraction: float = TRANSIENT_TRIM_FRACTION,
) -> np.ndarray:
    """Remove the initial transient portion of a signal.

    The first *fraction* of samples are dropped to skip the ramp-up phase
    when the participant starts walking. Uses floor division — any remainder
    is preserved at the signal start.

    Args:
        signal: 1-D signal array.
        fraction: Fraction of samples to remove from the start (default 0.20).
            Must be in [0, 1).

    Returns:
        Trimmed signal.

    Raises:
        ValueError: If *fraction* is not in [0, 1).
    """
    if not (0 <= fraction < 1):
        raise ValueError(f"Trim fraction must be in [0, 1), got {fraction}.")
    start = int(len(signal) * fraction)
    return signal[start:]


def center_signal(signal: np.ndarray) -> np.ndarray:
    """Center a ROM signal around zero.

    Subtracts the midpoint of the signal range so that (max + min) / 2 = 0.

    Args:
        signal: 1-D ROM angle array.

    Returns:
        Centered signal.

    Raises:
        ValueError: If *signal* is empty.
    """
    if signal.size == 0:
        raise ValueError("Cannot center signal: signal is empty.")
    return signal - (np.max(signal) + np.min(signal)) / 2


def preprocess_rom(signal: np.ndarray) -> np.ndarray:
    """Apply the full ROM preprocessing chain.

    Steps applied in order:
    1. Zero-phase high-pass Butterworth filter (``highpass_filter``) — applied at
       the original sample rate (FS_ROM ≈ 11.5 Hz) so the normalized cutoff of 0.1
       corresponds to ~0.575 Hz, removing only DC drift while preserving the gait
       signal (~0.5–2 Hz). Filtering after upsampling would shift the cutoff to
       ~5.75 Hz and destroy the signal.
    2. Upsample 10× (``interpolate_rom``)
    3. Drop initial transient (``trim_transient``)
    4. Center around zero (``center_signal``)

    Args:
        signal: Raw 1-D ROM angle array.

    Returns:
        Preprocessed signal ready for peak detection.
    """
    signal = highpass_filter(signal)  # filter at original FS_ROM before upsampling
    signal = interpolate_rom(signal)
    signal = trim_transient(signal)
    signal = center_signal(signal)
    return signal
