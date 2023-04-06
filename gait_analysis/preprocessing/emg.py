import numpy as np
import pandas as pd
from scipy.signal import resample

from gait_analysis.config import INTERPOLATION_FACTOR, IQR_FACTOR


def remove_outliers(series: pd.Series) -> pd.Series:
    """Remove outliers from an EMG series using the IQR method.

    Values outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR] are replaced with NaN.

    Args:
        series: Raw EMG amplitude series.

    Returns:
        Series with outlier values set to NaN.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_FACTOR * iqr
    upper = q3 + IQR_FACTOR * iqr
    return series.where((series >= lower) & (series <= upper))


def resample_emg(signal: np.ndarray, upsample_factor: int = INTERPOLATION_FACTOR) -> np.ndarray:
    """Upsample an EMG signal using Fourier-based resampling.

    Args:
        signal: 1-D EMG amplitude array.
        upsample_factor: Multiplicative upsampling factor (default 10×).

    Returns:
        Upsampled array of length ``len(signal) * upsample_factor``.
    """
    return resample(signal, len(signal) * upsample_factor)


def detect_contractions(signal: np.ndarray) -> np.ndarray:
    """Create a binary contraction mask: 1 where signal exceeds its mean.

    Args:
        signal: 1-D EMG amplitude array. Must not be all-NaN.

    Returns:
        Integer array of 0s and 1s, same length as *signal*.

    Raises:
        ValueError: If *signal* is entirely NaN (threshold cannot be computed).
    """
    threshold = np.nanmean(signal)
    if np.isnan(threshold):
        raise ValueError("Cannot detect contractions: signal is entirely NaN.")
    return (signal > threshold).astype(int)


def replace_nan_with_mean(signal: np.ndarray) -> np.ndarray:
    """Replace NaN values with the array mean.

    Args:
        signal: 1-D array possibly containing NaNs.

    Returns:
        Array with NaNs replaced by the mean of non-NaN values.

    Raises:
        ValueError: If *signal* is entirely NaN (no valid mean can be computed).
    """
    if np.all(np.isnan(signal)):
        raise ValueError("Cannot replace NaNs: signal is entirely NaN.")
    mean = np.nanmean(signal)
    out = signal.copy()
    out[np.isnan(out)] = mean
    return out
