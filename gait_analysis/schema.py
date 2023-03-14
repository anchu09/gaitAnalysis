"""
Column and key name constants shared across all modules.

Import from here — never hardcode these strings in analysis or visualization code.
"""

# Results DataFrame columns
PATIENT = "patient"
VELOCITY = "velocity"
WEIGHT_SUPPORT = "weight_support"

# ROM metrics
ROM_SWING_LEFT = "rom_swing_left"
ROM_STANCE_LEFT = "rom_stance_left"
ROM_SWING_RIGHT = "rom_swing_right"
ROM_STANCE_RIGHT = "rom_stance_right"

# EMG metrics
MAD_TIBIALIS_LEFT = "mad_tibialis_left"
MAD_TIBIALIS_RIGHT = "mad_tibialis_right"
MAD_GASTROCNEMIUS_LEFT = "mad_gastrocnemius_left"
MAD_GASTROCNEMIUS_RIGHT = "mad_gastrocnemius_right"
MEAN_EMG_TIBIALIS_LEFT = "mean_emg_tibialis_left"
MEAN_EMG_TIBIALIS_RIGHT = "mean_emg_tibialis_right"
MEAN_EMG_GASTROCNEMIUS_LEFT = "mean_emg_gastrocnemius_left"
MEAN_EMG_GASTROCNEMIUS_RIGHT = "mean_emg_gastrocnemius_right"
CI_LEFT = "coactivation_index_left"
CI_RIGHT = "coactivation_index_right"

# Gait metrics
N_STEPS = "n_steps"
DURATION_S = "duration_s"
CADENCE = "cadence_steps_per_min"
SLS_LEFT = "sls_left_pct"
SLS_RIGHT = "sls_right_pct"
DLS = "dls_pct"
SYMMETRY_ROM = "symmetry_rom"
ROM_RANGE_LEFT = "rom_range_left"
ROM_RANGE_RIGHT = "rom_range_right"

# Gait phase keys (used in phase dicts)
SWING = "swing"
STANCE = "stance"

# Gait phase peak type tags (used in gait_cycle structured arrays)
PEAK_TYPE_MAX: int = 1
PEAK_TYPE_MIN: int = -1

# Raw EMG signal column names as they appear after loading (must match EMG_COLS keys)
EMG_RMS_LEFT_TIBIALIS = "rms_left_tibialis"
EMG_RMS_RIGHT_TIBIALIS = "rms_right_tibialis"
EMG_RMS_LEFT_GASTROCNEMIUS = "rms_left_gastrocnemius"
EMG_RMS_RIGHT_GASTROCNEMIUS = "rms_right_gastrocnemius"

# Preprocessed ROM column names as they appear after loading (must match ROM_COLS keys)
ROM_LEFT_HIP = "left_hip"
ROM_RIGHT_HIP = "right_hip"

# Per-side lookup maps: (side, muscle) → raw EMG signal key
EMG_SIGNAL_KEY: dict[tuple[str, str], str] = {
    ("left", "tibialis"): EMG_RMS_LEFT_TIBIALIS,
    ("right", "tibialis"): EMG_RMS_RIGHT_TIBIALIS,
    ("left", "gastrocnemius"): EMG_RMS_LEFT_GASTROCNEMIUS,
    ("right", "gastrocnemius"): EMG_RMS_RIGHT_GASTROCNEMIUS,
}

# Per-side lookup maps: (side, phase) → results DataFrame column
ROM_MEAN_ANGLE_KEY: dict[tuple[str, str], str] = {
    ("left", SWING): ROM_SWING_LEFT,
    ("left", STANCE): ROM_STANCE_LEFT,
    ("right", SWING): ROM_SWING_RIGHT,
    ("right", STANCE): ROM_STANCE_RIGHT,
}

MAD_KEY: dict[tuple[str, str], str] = {
    ("left", "tibialis"): MAD_TIBIALIS_LEFT,
    ("right", "tibialis"): MAD_TIBIALIS_RIGHT,
    ("left", "gastrocnemius"): MAD_GASTROCNEMIUS_LEFT,
    ("right", "gastrocnemius"): MAD_GASTROCNEMIUS_RIGHT,
}

MEAN_EMG_KEY: dict[tuple[str, str], str] = {
    ("left", "tibialis"): MEAN_EMG_TIBIALIS_LEFT,
    ("right", "tibialis"): MEAN_EMG_TIBIALIS_RIGHT,
    ("left", "gastrocnemius"): MEAN_EMG_GASTROCNEMIUS_LEFT,
    ("right", "gastrocnemius"): MEAN_EMG_GASTROCNEMIUS_RIGHT,
}

CI_KEY: dict[str, str] = {
    "left": CI_LEFT,
    "right": CI_RIGHT,
}

SLS_KEY: dict[str, str] = {
    "left": SLS_LEFT,
    "right": SLS_RIGHT,
}
