"""Tests for data loading and validation."""

import numpy as np
import pandas as pd
import pytest

from gait_analysis.config import EMG_COLS, ROM_COLS
from gait_analysis.data.loader import load_emg_files, load_rom_files
from gait_analysis.data.validator import validate_emg_dataframe, validate_rom_dataframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def emg_dir(tmp_path):
    """Minimal valid EMG CSV: 11 semicolon-delimited columns, comma decimal."""
    n_cols = max(EMG_COLS.values()) + 1  # = 11
    data = [[float(i + 1)] * n_cols for i in range(5)]
    df = pd.DataFrame(data)
    content = df.to_csv(sep=";", decimal=",", index=False)
    (tmp_path / "patient_alta00.csv").write_text(content)
    return tmp_path


@pytest.fixture
def rom_dir(tmp_path):
    """Minimal valid ROM XLSX: 6 columns (indices 3-5 are the ROM columns)."""
    data = [[0.0, 0.0, 0.0, 15.0, -10.0, 45.0] for _ in range(10)]
    df = pd.DataFrame(data)
    df.to_excel(tmp_path / "patient_alta00.xlsx", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# data.loader
# ---------------------------------------------------------------------------


def test_load_emg_files_returns_correct_columns(emg_dir):
    result = load_emg_files(emg_dir)
    assert len(result) == 1
    assert set(result["patient_alta00"].columns) == set(EMG_COLS.keys())


def test_load_emg_files_values_are_extracted(emg_dir):
    result = load_emg_files(emg_dir)
    # All values come from col index 7-10 which were set to float(i+1) = 1..5
    assert result["patient_alta00"].notna().all().all()


def test_load_emg_files_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_emg_files(tmp_path / "nonexistent")


def test_load_emg_files_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No CSV"):
        load_emg_files(tmp_path)


def test_load_rom_files_returns_correct_columns(rom_dir):
    result = load_rom_files(rom_dir)
    assert len(result) == 1
    assert set(result["patient_alta00"].columns) == set(ROM_COLS.keys())


def test_load_rom_files_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rom_files(tmp_path / "nonexistent")


def test_load_rom_files_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No XLSX"):
        load_rom_files(tmp_path)


# ---------------------------------------------------------------------------
# data.validator — EMG
# ---------------------------------------------------------------------------


def test_validate_emg_valid():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.5, 1.5]})
    assert validate_emg_dataframe(df, "trial") is True


def test_validate_emg_empty():
    assert validate_emg_dataframe(pd.DataFrame(), "trial") is False


def test_validate_emg_all_nan_column():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    assert validate_emg_dataframe(df, "trial") is False


def test_validate_emg_too_many_nans():
    # 70% NaN in column "a" — exceeds the 60% threshold
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 2.0]})
    assert validate_emg_dataframe(df, "trial") is False


def test_validate_emg_negative_values():
    df = pd.DataFrame({"a": [1.0, -0.5, 2.0]})
    assert validate_emg_dataframe(df, "trial") is False


# ---------------------------------------------------------------------------
# data.validator — ROM
# ---------------------------------------------------------------------------


def test_validate_rom_valid():
    df = pd.DataFrame({"left_hip": [10.0, 15.0], "right_hip": [-5.0, -10.0]})
    assert validate_rom_dataframe(df, "trial") is True


def test_validate_rom_empty():
    assert validate_rom_dataframe(pd.DataFrame(), "trial") is False


def test_validate_rom_all_nan_column():
    df = pd.DataFrame({"left_hip": [10.0, 15.0], "right_hip": [np.nan, np.nan]})
    assert validate_rom_dataframe(df, "trial") is False


def test_validate_rom_too_many_nans():
    nans = [np.nan] * 7
    df = pd.DataFrame({"left_hip": [1.0] + nans, "right_hip": [1.0] + nans})
    assert validate_rom_dataframe(df, "trial") is False
