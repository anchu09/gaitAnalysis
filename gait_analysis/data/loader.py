import logging
import re
from pathlib import Path

import pandas as pd

from gait_analysis.config import EMG_COLS, ROM_COLS

logger = logging.getLogger(__name__)

# Expected filename pattern: <patient>_<velocity>_<weight>.csv / .xlsx
# Real filenames use no separator between velocity and weight: alta00, baja25
_FILENAME_PATTERN = re.compile(
    r"^(?P<patient>.+)_(?P<velocity>baja|medi|alta)(?P<weight>\d+)$",
    re.IGNORECASE,
)


def parse_filename(fname: str) -> tuple[str, str, str] | None:
    """Extract (patient, velocity, weight_support) from a trial filename.

    Args:
        fname: Filename stem (no extension).

    Returns:
        Tuple of (patient, velocity, weight_support) strings, or None if the
        filename does not match the expected convention.
    """
    m = _FILENAME_PATTERN.match(fname)
    if not m:
        return None
    return m.group("patient"), m.group("velocity"), m.group("weight")


def load_emg_files(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load all EMG CSV files from *data_dir*.

    Each file is semicolon-delimited with a comma as the decimal separator.
    Columns at indices defined in :data:`~gait_analysis.config.EMG_COLS` are
    extracted and renamed to their semantic muscle names.

    Args:
        data_dir: Directory containing EMG CSV files.

    Returns:
        Dict mapping filename stem → DataFrame with renamed muscle columns.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        ValueError: If no CSV files are found, or if any file is missing the
            expected column indices defined in ``EMG_COLS``.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"EMG directory not found: {data_dir}")

    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {data_dir}")

    muscle_cols: dict[int, str] = {idx: name for name, idx in EMG_COLS.items()}
    result: dict[str, pd.DataFrame] = {}

    for f in files:
        try:
            raw = pd.read_csv(f, sep=";", decimal=",", header=0)
        except (OSError, pd.errors.ParserError) as exc:
            raise ValueError(f"Failed to parse EMG file '{f.name}': {exc}") from exc

        missing = [i for i in muscle_cols if i >= len(raw.columns)]
        if missing:
            raise ValueError(
                f"EMG file '{f.name}' has {len(raw.columns)} columns but "
                f"EMG_COLS expects indices {missing}."
            )

        df = raw.iloc[:, list(muscle_cols)].copy()
        df.columns = pd.Index([muscle_cols[i] for i in muscle_cols])
        result[f.stem] = df

    logger.info("Loaded %d EMG files from %s", len(result), data_dir)
    return result


def load_rom_files(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load all ROM XLSX files from *data_dir*.

    Columns at indices defined in :data:`~gait_analysis.config.ROM_COLS` are
    extracted and renamed to their semantic joint names.

    Args:
        data_dir: Directory containing ROM XLSX files.

    Returns:
        Dict mapping filename stem → DataFrame with renamed joint columns.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        ValueError: If no XLSX files are found, or if any file is missing the
            expected column indices defined in ``ROM_COLS``.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"ROM directory not found: {data_dir}")

    files = sorted(data_dir.glob("*.xlsx"))
    if not files:
        raise ValueError(f"No XLSX files found in {data_dir}")

    joint_cols: dict[int, str] = {idx: name for name, idx in ROM_COLS.items()}
    result: dict[str, pd.DataFrame] = {}

    for f in files:
        try:
            raw = pd.read_excel(f, header=0)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Failed to parse ROM file '{f.name}': {exc}") from exc

        missing = [i for i in joint_cols if i >= len(raw.columns)]
        if missing:
            raise ValueError(
                f"ROM file '{f.name}' has {len(raw.columns)} columns but "
                f"ROM_COLS expects indices {missing}."
            )

        df = raw.iloc[:, list(joint_cols)].copy()
        df.columns = pd.Index([joint_cols[i] for i in joint_cols])
        result[f.stem] = df

    logger.info("Loaded %d ROM files from %s", len(result), data_dir)
    return result
