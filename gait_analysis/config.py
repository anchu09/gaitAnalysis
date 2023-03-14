"""
All pipeline constants. Edit here — do not scatter magic numbers in analysis modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Sampling rates
# ---------------------------------------------------------------------------
FS_EMG: float = 4.03  # Hz — post-envelope (0.25 s intervals)
FS_ROM: float = 11.5  # Hz — measured from the SWalker device output (verified across all
# trials: 484 ± 2 samples over ~42 s). If a new device or recording setup is used,
# re-derive this value from a known-duration trial before running the pipeline.
INTERPOLATION_FACTOR: int = 10  # Both signals upsampled ×10 before alignment

# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------
IQR_FACTOR: float = 1.5  # Outlier threshold multiplier
TRANSIENT_TRIM_FRACTION: float = 0.20  # Drop first 20 % of each trial

# Butterworth high-pass filter applied to ROM
FILTER_ORDER: int = 4
FILTER_CUTOFF: float = 0.1  # Normalized frequency [0, 1]

# ---------------------------------------------------------------------------
# Gait cycle detection
# ---------------------------------------------------------------------------
PEAK_MIN_DISTANCE: int = 10  # Minimum samples between consecutive peaks

# ---------------------------------------------------------------------------
# Trial conditions
# ---------------------------------------------------------------------------
VELOCITIES: tuple[str, ...] = ("baja", "medi", "alta")
WEIGHT_SUPPORTS: tuple[int, ...] = (0, 25, 50)

# Display labels for plots — maps internal file-based keys to readable English
VELOCITY_LABELS: dict[str, str] = {"baja": "Low", "medi": "Medium", "alta": "High"}
WEIGHT_SUPPORT_LABELS: dict[int, str] = {0: "0% BWS", 25: "25% BWS", 50: "50% BWS"}

# ---------------------------------------------------------------------------
# Raw EMG column indices (0-based, after CSV parse)
# ---------------------------------------------------------------------------
EMG_COLS: dict[str, int] = {
    "rms_right_gastrocnemius": 7,
    "rms_right_tibialis": 8,
    "rms_left_gastrocnemius": 9,
    "rms_left_tibialis": 10,
}

# ---------------------------------------------------------------------------
# Raw ROM column indices (0-based, after XLSX parse)
# ---------------------------------------------------------------------------
ROM_COLS: dict[str, int] = {
    "left_hip": 3,  # "Left Hip Real"
    "right_hip": 4,  # "Right Hip Real"
    "weight_gauge": 5,
}

# ---------------------------------------------------------------------------
# Default data directories (override via run_pipeline arguments)
# ---------------------------------------------------------------------------
DEFAULT_EMG_DIR: Path = Path("data/EMG")
DEFAULT_ROM_DIR: Path = Path("data/ROM")
DEFAULT_OUTPUT_DIR: Path = Path("results")
