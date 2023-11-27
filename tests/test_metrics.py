import numpy as np
import pandas as pd
import pytest

from gait_analysis.analysis.emg_metrics import (
    compute_coactivation_index,
    compute_mad,
    compute_mean_amplitude,
)
from gait_analysis.analysis.gait_cycle import (
    build_peak_sequence,
    correct_peak_artifacts,
    detect_peaks,
    extract_phases,
)
from gait_analysis.analysis.gait_metrics import compute_cadence, compute_sls_dls, compute_symmetry
from gait_analysis.analysis.rom_metrics import compute_rom_mean_angle
from gait_analysis.data.loader import parse_filename
from gait_analysis.preprocessing.emg import detect_contractions, remove_outliers

# ---------------------------------------------------------------------------
# preprocessing.emg
# ---------------------------------------------------------------------------


def test_remove_outliers_removes_extremes():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    result = remove_outliers(s)
    assert result.isna().sum() == 1
    assert pd.isna(result.iloc[4])


def test_remove_outliers_keeps_normal_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = remove_outliers(s)
    assert result.isna().sum() == 0


def test_detect_contractions_binary_output():
    signal = np.array([0.1, 0.5, 0.8, 0.2, 0.9])
    result = detect_contractions(signal)
    assert set(result).issubset({0, 1})
    assert result.dtype in (np.int32, np.int64, int)


def test_detect_contractions_threshold_is_mean():
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    result = detect_contractions(signal)
    # mean = 2.5; values > 2.5 → 1
    np.testing.assert_array_equal(result, [0, 0, 1, 1])


def test_detect_contractions_all_nan_raises():
    signal = np.array([np.nan, np.nan])
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="entirely NaN"):
        detect_contractions(signal)


# ---------------------------------------------------------------------------
# analysis.emg_metrics
# ---------------------------------------------------------------------------


def test_coactivation_index_identical_signals():
    signal = np.array([1.0, 2.0, 3.0])
    ci = compute_coactivation_index(signal, signal.copy())
    assert ci == pytest.approx(100.0)


def test_coactivation_index_zero_antagonist():
    ag = np.array([1.0, 2.0, 3.0])
    antag = np.array([0.0, 0.0, 0.0])
    ci = compute_coactivation_index(ag, antag)
    assert ci == pytest.approx(0.0)


def test_coactivation_index_both_zero_returns_nan():
    result = compute_coactivation_index(np.zeros(3), np.zeros(3))
    assert np.isnan(result)


def test_compute_mad_all_above_mean():
    # All values equal → none strictly above mean → active integral = 0 → MAD = 0
    signal = np.array([1.0, 1.0, 1.0])
    result = compute_mad(signal)
    assert result == pytest.approx(0.0)


def test_compute_mad_half_active():
    signal = np.array([1.0, 1.0, 3.0, 3.0])
    result = compute_mad(signal)
    # mean = 2.0; active = [3, 3] → integral 6 / total 8 = 75%
    assert result == pytest.approx(75.0)


def test_compute_mad_zero_signal_raises():
    with pytest.raises(ValueError, match="zero"):
        compute_mad(np.zeros(5))


def test_compute_mean_amplitude_empty_phases_raises():
    signal = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="empty"):
        compute_mean_amplitude(signal, [])


def test_compute_mean_amplitude_single_phase():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_mean_amplitude(signal, [(1, 4)])
    assert result == pytest.approx(3.0)  # mean of [2, 3, 4]


# ---------------------------------------------------------------------------
# analysis.rom_metrics
# ---------------------------------------------------------------------------


def test_compute_rom_mean_angle_empty_phases_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_rom_mean_angle(np.ones(10), [])


def test_compute_rom_mean_angle_single_phase():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_rom_mean_angle(signal, [(1, 4)])
    assert result == pytest.approx(3.0)  # mean of [2, 3, 4]


# ---------------------------------------------------------------------------
# analysis.gait_cycle
# ---------------------------------------------------------------------------


def test_build_peak_sequence_alternating():
    maxima = np.array([10, 30, 50])
    minima = np.array([20, 40, 60])
    seq = build_peak_sequence(maxima, minima)
    types = seq["type"].tolist()
    for i in range(len(types) - 1):
        assert types[i] != types[i + 1], f"Non-alternating at index {i}: {types}"


def test_build_peak_sequence_sorted_by_index():
    maxima = np.array([50, 10])
    minima = np.array([30, 70])
    seq = build_peak_sequence(maxima, minima)
    indices = seq["index"].tolist()
    assert indices == sorted(indices)


def test_extract_phases_phase_counts_match_cycle_geometry():
    """Two full sine cycles starting at a maximum yield 2 stance and 1 swing phase.

    Peak sequence for sin(0..4π): max@~50, min@~150, max@~250, min@~350.
    Consecutive pairs: max→min (stance), min→max (swing), max→min (stance).
    """
    signal = np.sin(np.linspace(0, 4 * np.pi, 400))
    maxima, minima = detect_peaks(signal, min_distance=5)
    seq = build_peak_sequence(maxima, minima)
    seq = correct_peak_artifacts(signal, seq)
    phases = extract_phases(signal, seq)
    assert len(phases["stance"]) == 2
    assert len(phases["swing"]) == 1
    for s, e in phases["stance"] + phases["swing"]:
        assert s < e, f"Invalid phase interval: start {s} >= end {e}"


def test_extract_phases_too_short_raises():
    signal = np.array([1.0, 2.0, 1.0])
    seq = build_peak_sequence(np.array([1]), np.array([]))
    with pytest.raises(ValueError):
        extract_phases(signal, seq)


# ---------------------------------------------------------------------------
# analysis.gait_metrics
# ---------------------------------------------------------------------------


def test_cadence_calculation():
    result = compute_cadence(n_steps=60, duration_s=60.0)
    assert result == pytest.approx(60.0)


def test_cadence_zero_duration_raises():
    with pytest.raises(ValueError, match="positive"):
        compute_cadence(10, 0.0)


def test_cadence_negative_duration_raises():
    with pytest.raises(ValueError, match="positive"):
        compute_cadence(10, -1.0)


def test_symmetry_perfect():
    assert compute_symmetry(5.0, 5.0) == pytest.approx(0.0)


def test_symmetry_asymmetric():
    result = compute_symmetry(6.0, 4.0)
    # |6-4| / 5 * 100 = 40
    assert result == pytest.approx(40.0)


def test_symmetry_both_zero_returns_nan():
    assert np.isnan(compute_symmetry(0.0, 0.0))


def test_sls_dls_no_overlap():
    # Left in stance 0–5, right in stance 5–10, no overlap.
    left = {"stance": [(0, 5)], "swing": [(5, 10)]}
    right = {"stance": [(5, 10)], "swing": [(0, 5)]}
    sls_l, sls_r, dls = compute_sls_dls(left, right, 10)
    assert sls_l == pytest.approx(50.0)
    assert sls_r == pytest.approx(50.0)
    assert dls == pytest.approx(0.0)


def test_sls_dls_full_overlap():
    # Both legs in stance the entire time: pure DLS, no SLS.
    left = {"stance": [(0, 10)]}
    right = {"stance": [(0, 10)]}
    sls_l, sls_r, dls = compute_sls_dls(left, right, 10)
    assert sls_l == pytest.approx(0.0)
    assert sls_r == pytest.approx(0.0)
    assert dls == pytest.approx(100.0)


def test_sls_dls_partial_overlap():
    # Left stance 0–6, right stance 4–10, total = 10.
    # sls_left  (0–4)  = 4 samples → 40%
    # dls       (4–6)  = 2 samples → 20%
    # sls_right (6–10) = 4 samples → 40%
    left = {"stance": [(0, 6)]}
    right = {"stance": [(4, 10)]}
    sls_l, sls_r, dls = compute_sls_dls(left, right, 10)
    assert sls_l == pytest.approx(40.0)
    assert sls_r == pytest.approx(40.0)
    assert dls == pytest.approx(20.0)


def test_sls_dls_missing_stance_key_raises():
    with pytest.raises(ValueError, match="stance"):
        compute_sls_dls({"swing": [(0, 5)]}, {"stance": [(0, 5)]}, 10)


def test_sls_dls_zero_total_samples_raises():
    with pytest.raises(ValueError, match="positive"):
        compute_sls_dls({"stance": [(0, 5)]}, {"stance": [(0, 5)]}, 0)


# ---------------------------------------------------------------------------
# data.loader
# ---------------------------------------------------------------------------


def test_parse_filename_valid():
    # Real filename format: <patient>_<velocity><weight> (no underscore before weight)
    result = parse_filename("JohnDoe_alta25")
    assert result == ("JohnDoe", "alta", "25")


def test_parse_filename_invalid_returns_none():
    assert parse_filename("random_file_name") is None
