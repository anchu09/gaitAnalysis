"""
Orchestrates the full EMG/ROM analysis pipeline.

Usage::

    from pathlib import Path
    from gait_analysis.pipeline import run_pipeline

    results = run_pipeline(
        emg_dir=Path("data/EMG"),
        rom_dir=Path("data/ROM"),
        output_dir=Path("results"),
    )
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import gait_analysis.schema as sc
from gait_analysis.analysis.emg_metrics import compute_coactivation_index, compute_mad
from gait_analysis.analysis.gait_cycle import (
    build_peak_sequence,
    correct_peak_artifacts,
    detect_peaks,
    extract_phases,
)
from gait_analysis.analysis.gait_metrics import compute_cadence, compute_sls_dls, compute_symmetry
from gait_analysis.analysis.rom_metrics import compute_rom_mean_angle, compute_rom_range
from gait_analysis.config import EMG_COLS, FS_ROM, INTERPOLATION_FACTOR
from gait_analysis.data.loader import load_emg_files, load_rom_files, parse_filename
from gait_analysis.data.validator import validate_emg_dataframe, validate_rom_dataframe
from gait_analysis.preprocessing.emg import remove_outliers, replace_nan_with_mean, resample_emg
from gait_analysis.preprocessing.rom import preprocess_rom

logger = logging.getLogger(__name__)

# Sides processed in every trial
_SIDES = ("left", "right")
_HIP_COL = {"left": sc.ROM_LEFT_HIP, "right": sc.ROM_RIGHT_HIP}


def _preprocess_emg(
    emg_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Outlier-remove and upsample all four EMG channels.

    Args:
        emg_df: DataFrame with columns matching ``EMG_COLS`` keys.

    Returns:
        Dict mapping EMG signal name → upsampled numpy array.
        Only columns actually present in *emg_df* are included.
    """
    result: dict[str, np.ndarray] = {}
    for col_name in EMG_COLS:
        if col_name not in emg_df.columns:
            logger.warning("EMG column '%s' not found — skipping.", col_name)
            continue
        # Fill outlier-NaNs before resampling — FFT-based resample propagates NaN
        cleaned = replace_nan_with_mean(remove_outliers(emg_df[col_name]).values)
        result[col_name] = resample_emg(cleaned)
    return result


def _preprocess_and_segment_rom(
    rom_df: pd.DataFrame,
    key: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[tuple[int, int]]]]]:
    """Preprocess ROM signals and extract gait cycle phases for both hips.

    Args:
        rom_df: DataFrame with columns matching ``ROM_COLS`` keys.
        key: Trial key (used only in warning messages).

    Returns:
        Tuple of:
        - preprocessed: dict mapping hip column name → preprocessed numpy array
        - phases: dict mapping hip column name → {'swing': [...], 'stance': [...]}

        Only sides for which preprocessing and phase detection succeed are included.
    """
    preprocessed: dict[str, np.ndarray] = {}
    phases: dict[str, dict[str, list[tuple[int, int]]]] = {}

    for hip_col in (sc.ROM_LEFT_HIP, sc.ROM_RIGHT_HIP):
        if hip_col not in rom_df.columns:
            logger.warning("ROM column '%s' not found in '%s' — skipping side.", hip_col, key)
            continue
        raw = rom_df[hip_col].dropna().values
        if raw.size < 10:
            logger.warning(
                "ROM signal '%s/%s' has only %d samples — too short, skipping.",
                key,
                hip_col,
                raw.size,
            )
            continue
        try:
            proc = preprocess_rom(raw)
            maxima, minima = detect_peaks(proc)
            sequence = build_peak_sequence(maxima, minima)
            sequence = correct_peak_artifacts(proc, sequence)
            phases[hip_col] = extract_phases(proc, sequence)
            preprocessed[hip_col] = proc
        except ValueError as exc:
            logger.warning("ROM processing failed for '%s/%s': %s", key, hip_col, exc)

    return preprocessed, phases


def _compute_emg_metrics_for_side(
    emg: dict[str, np.ndarray],
    side: str,
) -> dict[str, float]:
    """Compute MAD, mean amplitude, and CI for one body side.

    Args:
        emg: Upsampled EMG signals keyed by signal name (from ``EMG_COLS``).
        side: ``'left'`` or ``'right'``.

    Returns:
        Dict of metric name → value for this side. Missing muscles are omitted.
    """
    row: dict[str, float] = {}
    tib_key = sc.EMG_SIGNAL_KEY[(side, "tibialis")]
    gas_key = sc.EMG_SIGNAL_KEY[(side, "gastrocnemius")]

    for muscle in ("tibialis", "gastrocnemius"):
        sig_key = sc.EMG_SIGNAL_KEY[(side, muscle)]
        if sig_key not in emg:
            continue
        sig = emg[sig_key]
        try:
            row[sc.MAD_KEY[(side, muscle)]] = compute_mad(sig)
        except ValueError as exc:
            logger.warning("MAD computation failed for %s %s: %s", side, muscle, exc)
        row[sc.MEAN_EMG_KEY[(side, muscle)]] = float(sig.mean())

    if tib_key in emg and gas_key in emg:
        row[sc.CI_KEY[side]] = compute_coactivation_index(emg[tib_key], emg[gas_key])

    return row


def _compute_rom_metrics_for_side(
    preprocessed: dict[str, np.ndarray],
    phases: dict[str, dict[str, list[tuple[int, int]]]],
    side: str,
) -> dict[str, float]:
    """Compute ROM amplitude (per phase) and peak-to-peak range for one body side.

    Args:
        preprocessed: Preprocessed ROM signals keyed by hip column name.
        phases: Gait cycle phases keyed by hip column name.
        side: ``'left'`` or ``'right'``.

    Returns:
        Dict of metric name → value. Empty if the side's hip signal is missing.
    """
    row: dict[str, float] = {}
    hip_col = _HIP_COL[side]

    if hip_col not in preprocessed or hip_col not in phases:
        return row

    sig = preprocessed[hip_col]
    phi = phases[hip_col]

    for phase_name in (sc.SWING, sc.STANCE):
        metric_key = sc.ROM_MEAN_ANGLE_KEY[(side, phase_name)]
        try:
            row[metric_key] = compute_rom_mean_angle(sig, phi.get(phase_name, []))
        except ValueError as exc:
            logger.warning("ROM amplitude failed for %s %s: %s", side, phase_name, exc)

    range_key = sc.ROM_RANGE_LEFT if side == "left" else sc.ROM_RANGE_RIGHT
    try:
        row[range_key] = compute_rom_range(sig)
    except ValueError as exc:
        logger.warning("ROM range failed for %s: %s", side, exc)

    return row


def _compute_gait_metrics(
    preprocessed: dict[str, np.ndarray],
    phases: dict[str, dict[str, list[tuple[int, int]]]],
) -> dict[str, float]:
    """Compute cadence, SLS/DLS, and ROM symmetry from both hip signals.

    Args:
        preprocessed: Preprocessed ROM signals keyed by hip column name.
        phases: Gait cycle phases keyed by hip column name.

    Returns:
        Dict of gait metric name → value.
    """
    row: dict[str, float] = {}
    left_col = sc.ROM_LEFT_HIP
    right_col = sc.ROM_RIGHT_HIP

    if left_col not in preprocessed:
        return row

    left_sig = preprocessed[left_col]
    duration_s = len(left_sig) / (FS_ROM * INTERPOLATION_FACTOR)

    # One swing phase per leg = one step; sum both legs for total step count
    n_steps = sum(len(phases[h].get(sc.SWING, [])) for h in (left_col, right_col) if h in phases)
    row[sc.N_STEPS] = n_steps
    row[sc.DURATION_S] = duration_s

    try:
        row[sc.CADENCE] = compute_cadence(n_steps, duration_s)
    except ValueError as exc:
        logger.warning("Cadence computation failed: %s", exc)

    if left_col in phases and right_col in phases:
        total = len(left_sig)
        try:
            sls_l, sls_r, dls = compute_sls_dls(phases[left_col], phases[right_col], total)
            row[sc.SLS_LEFT] = sls_l
            row[sc.SLS_RIGHT] = sls_r
            row[sc.DLS] = dls
        except ValueError as exc:
            logger.warning("SLS/DLS computation failed: %s", exc)

    return row


def _process_case(
    key: str,
    emg_df: pd.DataFrame,
    rom_df: pd.DataFrame,
) -> dict | None:
    """Process one trial (patient × velocity × weight_support).

    Args:
        key: Filename stem — must match the naming convention
            ``<patient>_<velocity>_<weight_support>``.
        emg_df: EMG DataFrame for this trial.
        rom_df: ROM DataFrame for this trial.

    Returns:
        Dict of all computed metrics, or ``None`` if the trial cannot be
        processed (invalid filename, failed validation).
    """
    if not validate_emg_dataframe(emg_df, key) or not validate_rom_dataframe(rom_df, key):
        logger.warning("Validation failed for trial '%s' — skipping.", key)
        return None

    parsed = parse_filename(key)
    if parsed is None:
        logger.warning(
            "Filename '%s' does not match the expected convention "
            "<patient>_<velocity>_<weight_support> — skipping.",
            key,
        )
        return None
    patient, velocity, weight_support = parsed

    emg = _preprocess_emg(emg_df)
    preprocessed, phases = _preprocess_and_segment_rom(rom_df, key)

    row: dict = {
        sc.PATIENT: patient,
        sc.VELOCITY: velocity,
        sc.WEIGHT_SUPPORT: int(weight_support),
    }

    for side in _SIDES:
        row.update(_compute_emg_metrics_for_side(emg, side))
        row.update(_compute_rom_metrics_for_side(preprocessed, phases, side))

    row.update(_compute_gait_metrics(preprocessed, phases))

    # ROM symmetry (only when both sides have swing amplitude)
    if sc.ROM_SWING_LEFT in row and sc.ROM_SWING_RIGHT in row:
        row[sc.SYMMETRY_ROM] = compute_symmetry(row[sc.ROM_SWING_LEFT], row[sc.ROM_SWING_RIGHT])

    return row


def run_pipeline(
    emg_dir: Path | str,
    rom_dir: Path | str,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Run the full EMG/ROM gait analysis pipeline.

    Loads all trial files, preprocesses signals, detects gait cycles, computes
    all metrics per trial, and assembles a results DataFrame.

    Args:
        emg_dir: Directory containing EMG CSV files.
        rom_dir: Directory containing ROM XLSX files.
        output_dir: If given, save ``results.csv`` to this directory.

    Returns:
        DataFrame with one row per successfully processed trial and all computed
        metrics as columns. Trials that fail validation or filename parsing are
        logged as warnings and excluded.
    """
    emg_data = load_emg_files(emg_dir)
    rom_data = load_rom_files(rom_dir)

    common_keys = sorted(set(emg_data) & set(rom_data))
    only_emg = set(emg_data) - set(rom_data)
    only_rom = set(rom_data) - set(emg_data)
    if only_emg:
        logger.warning("EMG files with no matching ROM: %s", sorted(only_emg))
    if only_rom:
        logger.warning("ROM files with no matching EMG: %s", sorted(only_rom))

    rows = []
    for key in tqdm(common_keys, desc="Processing trials"):
        row = _process_case(key, emg_data[key], rom_data[key])
        if row is not None:
            rows.append(row)

    results = pd.DataFrame(rows)
    logger.info("Pipeline complete: %d/%d trials processed.", len(results), len(common_keys))

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "results.csv"
        results.to_csv(csv_path, index=False)
        logger.info("Results saved to %s", csv_path)

    return results
