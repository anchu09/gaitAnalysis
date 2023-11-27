"""Tests for signal preprocessing modules."""

import numpy as np
import pytest

from gait_analysis.config import INTERPOLATION_FACTOR
from gait_analysis.preprocessing.emg import replace_nan_with_mean, resample_emg
from gait_analysis.preprocessing.rom import (
    center_signal,
    highpass_filter,
    interpolate_rom,
    preprocess_rom,
    trim_transient,
)

# ---------------------------------------------------------------------------
# preprocessing.emg
# ---------------------------------------------------------------------------


def test_replace_nan_with_mean_replaces_nans():
    signal = np.array([1.0, np.nan, 3.0])
    result = replace_nan_with_mean(signal)
    assert not np.any(np.isnan(result))
    assert result[1] == pytest.approx(2.0)  # mean of [1, 3]


def test_replace_nan_with_mean_no_nans_unchanged():
    signal = np.array([1.0, 2.0, 3.0])
    result = replace_nan_with_mean(signal)
    np.testing.assert_array_equal(result, signal)


def test_replace_nan_with_mean_all_nan_raises():
    with pytest.raises(ValueError, match="entirely NaN"):
        replace_nan_with_mean(np.array([np.nan, np.nan]))


def test_resample_emg_output_length():
    signal = np.ones(40)
    result = resample_emg(signal, upsample_factor=10)
    assert len(result) == 400


def test_resample_emg_default_factor():
    signal = np.ones(50)
    result = resample_emg(signal)
    assert len(result) == 50 * INTERPOLATION_FACTOR


# ---------------------------------------------------------------------------
# preprocessing.rom
# ---------------------------------------------------------------------------


def test_interpolate_rom_output_length():
    signal = np.random.default_rng(0).random(50)
    result = interpolate_rom(signal, upsample_factor=10)
    assert len(result) == 500


def test_interpolate_rom_default_factor():
    signal = np.ones(30)
    result = interpolate_rom(signal)
    assert len(result) == 30 * INTERPOLATION_FACTOR


def test_interpolate_rom_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        interpolate_rom(np.array([]))


def test_highpass_filter_output_length():
    signal = np.sin(np.linspace(0, 4 * np.pi, 200))
    result = highpass_filter(signal, cutoff=0.1, order=4)
    assert len(result) == len(signal)


def test_highpass_filter_attenuates_dc():
    # A constant signal is pure DC — high-pass filter should drive it to ~0
    signal = np.ones(200) * 5.0
    result = highpass_filter(signal, cutoff=0.1, order=4)
    assert np.abs(result).max() < 0.1


def test_highpass_filter_invalid_cutoff_raises():
    with pytest.raises(ValueError, match="cutoff"):
        highpass_filter(np.ones(100), cutoff=1.5)


def test_highpass_filter_invalid_order_raises():
    with pytest.raises(ValueError, match="order"):
        highpass_filter(np.ones(100), cutoff=0.1, order=0)


def test_trim_transient_removes_first_fraction():
    signal = np.arange(100, dtype=float)
    result = trim_transient(signal, fraction=0.20)
    assert len(result) == 80
    assert result[0] == pytest.approx(20.0)


def test_trim_transient_zero_fraction_unchanged():
    signal = np.arange(50, dtype=float)
    result = trim_transient(signal, fraction=0.0)
    np.testing.assert_array_equal(result, signal)


def test_trim_transient_invalid_fraction_raises():
    with pytest.raises(ValueError, match="fraction"):
        trim_transient(np.ones(50), fraction=1.0)


def test_center_signal_midpoint_is_zero():
    signal = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # midpoint = (0 + 8) / 2 = 4
    result = center_signal(signal)
    assert (np.max(result) + np.min(result)) == pytest.approx(0.0)


def test_center_signal_already_centered_unchanged():
    signal = np.array([-2.0, 0.0, 2.0])  # midpoint already 0
    result = center_signal(signal)
    np.testing.assert_array_almost_equal(result, signal)


def test_center_signal_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        center_signal(np.array([]))


def test_preprocess_rom_output_is_centered():
    # 100-sample sine wave — realistic minimal trial
    signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    result = preprocess_rom(signal)
    # After the full chain the midpoint must be zero
    assert (np.max(result) + np.min(result)) == pytest.approx(0.0, abs=1e-6)


def test_preprocess_rom_output_length():
    signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    result = preprocess_rom(signal)
    # trim(20%) → 80 samples → upsample(×10) → 800 samples
    assert len(result) == 800
