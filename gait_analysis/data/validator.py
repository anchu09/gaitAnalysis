import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Trials with more than this fraction of NaN are too sparse to process reliably.
_MAX_NAN_FRACTION: float = 0.60


def validate_emg_dataframe(df: pd.DataFrame, key: str) -> bool:
    """Check that an EMG DataFrame has the expected shape and signal quality.

    Validates:
    - DataFrame is not empty.
    - No column is entirely NaN.
    - No column exceeds 60 % NaN (signal too sparse).
    - No negative values (EMG RMS is always ≥ 0 by definition).

    Args:
        df: EMG DataFrame to validate.
        key: Filename stem used in log messages.

    Returns:
        True if valid, False otherwise (with a warning logged).
    """
    if df.empty:
        logger.warning("EMG DataFrame '%s' is empty.", key)
        return False

    all_nan = df.columns[df.isnull().all()].tolist()
    if all_nan:
        logger.warning("EMG '%s': columns %s are entirely NaN.", key, all_nan)
        return False

    nan_fraction = float(df.isnull().mean().max())
    if nan_fraction > _MAX_NAN_FRACTION:
        logger.warning(
            "EMG '%s': up to %.0f%% NaN values — trial too sparse to process.",
            key,
            nan_fraction * 100,
        )
        return False

    numeric = df.select_dtypes(include="number")
    if (numeric < 0).any().any():
        logger.warning("EMG '%s': negative values detected — RMS signal must be ≥ 0.", key)
        return False

    return True


def validate_rom_dataframe(df: pd.DataFrame, key: str) -> bool:
    """Check that a ROM DataFrame has the expected shape and signal quality.

    Validates:
    - DataFrame is not empty.
    - No column is entirely NaN.
    - No column exceeds 60 % NaN (signal too sparse).

    Args:
        df: ROM DataFrame to validate.
        key: Filename stem used in log messages.

    Returns:
        True if valid, False otherwise (with a warning logged).
    """
    if df.empty:
        logger.warning("ROM DataFrame '%s' is empty.", key)
        return False

    all_nan = df.columns[df.isnull().all()].tolist()
    if all_nan:
        logger.warning("ROM '%s': columns %s are entirely NaN.", key, all_nan)
        return False

    nan_fraction = float(df.isnull().mean().max())
    if nan_fraction > _MAX_NAN_FRACTION:
        logger.warning(
            "ROM '%s': up to %.0f%% NaN values — trial too sparse to process.",
            key,
            nan_fraction * 100,
        )
        return False

    return True
