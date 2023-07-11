from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample as sp_resample

from gait_analysis.schema import STANCE, SWING


def plot_rom_with_phases(
    signal: np.ndarray,
    phases: dict[str, list[tuple[int, int]]],
    side: str = "left",
    title: str | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot a ROM signal with swing and stance phases color-coded.

    Args:
        signal: 1-D preprocessed ROM angle array (degrees).
        phases: Dict with keys 'swing' and 'stance', each a list of (start, end).
        side: 'left' or 'right' — used for color scheme.
        title: Figure title (auto-generated if None).
        output_path: If given, save the figure here.

    Returns:
        Matplotlib Figure.
    """
    swing_color = "#4daf4a" if side == "left" else "#377eb8"
    stance_color = "#e41a1c" if side == "left" else "#984ea3"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal, color="0.3", linewidth=0.9, zorder=3)

    for s, e in phases.get(SWING, []):
        ax.axvspan(s, e, alpha=0.25, color=swing_color)
    for s, e in phases.get(STANCE, []):
        ax.axvspan(s, e, alpha=0.25, color=stance_color)

    swing_patch = mpatches.Patch(color=swing_color, alpha=0.4, label="Swing")
    stance_patch = mpatches.Patch(color=stance_color, alpha=0.4, label="Stance")
    ax.legend(handles=[swing_patch, stance_patch], fontsize=9)

    ax.set_xlabel("Samples")
    ax.set_ylabel("Hip angle (°)")
    ax.set_title(title or f"ROM — {side} hip", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_gait_cycle_normalized(
    left_signal: np.ndarray,
    right_signal: np.ndarray,
    left_phases: dict[str, list[tuple[int, int]]] | None = None,
    n_points: int = 100,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot both hip ROM signals normalized to a single gait cycle (0–100 %).

    Optionally shades the average double-limb support (DLS) and single-limb
    support (SLS) regions derived from the left leg phase timings, replicating
    the style of the paper's Fig. 4.

    Args:
        left_signal: Preprocessed left hip ROM array.
        right_signal: Preprocessed right hip ROM array.
        left_phases: Phase dict for the left leg. If given, the mean swing and
            stance boundaries are overlaid as shaded regions.
        n_points: Number of points in the normalized cycle (default 100).
        output_path: If given, save the figure here.

    Returns:
        Matplotlib Figure.
    """
    left_norm = sp_resample(left_signal, n_points)
    right_norm = sp_resample(right_signal, n_points)
    x = np.linspace(0, 100, n_points)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Shade gait phases using average phase boundaries from left leg
    if left_phases:
        n = len(left_signal)
        swing_intervals = left_phases.get(SWING, [])
        stance_intervals = left_phases.get(STANCE, [])

        if swing_intervals:
            # Convert sample indices to normalized percentage and average
            swing_starts = np.mean([s / n * 100 for s, _ in swing_intervals])
            swing_ends = np.mean([e / n * 100 for _, e in swing_intervals])
            ax.axvspan(swing_starts, swing_ends, alpha=0.12, color="#4daf4a", label="Swing (avg)")

        if stance_intervals:
            stance_starts = np.mean([s / n * 100 for s, _ in stance_intervals])
            stance_ends = np.mean([e / n * 100 for _, e in stance_intervals])
            ax.axvspan(
                stance_starts, stance_ends, alpha=0.12, color="#e41a1c", label="Stance (avg)"
            )

    ax.plot(x, left_norm, color="#e41a1c", linewidth=2.0, label="Left ROM signal")
    ax.plot(x, right_norm, color="#377eb8", linewidth=2.0, label="Right ROM signal")

    # Mark max flexion and max extension peaks
    left_max_idx = int(np.argmax(left_norm))
    left_min_idx = int(np.argmin(left_norm))
    ax.plot(x[left_max_idx], left_norm[left_max_idx], "o", color="#e41a1c", markersize=8, zorder=5)
    ax.plot(x[left_min_idx], left_norm[left_min_idx], "o", color="#377eb8", markersize=8, zorder=5)

    ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Percentage of gait cycle (%)")
    ax.set_ylabel("Sagittal hip ROM (°)")
    ax.set_title("Hip ROM over normalized gait cycle")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def _shade_mask(
    ax: plt.Axes,
    x: np.ndarray,
    mask: np.ndarray,
    color: str,
    label: str,
) -> None:
    """Fill contiguous True regions of *mask* as axvspan on *ax*."""
    in_region = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            ax.axvspan(x[start], x[i], alpha=0.3, color=color, linewidth=0)
            in_region = False
    if in_region:
        ax.axvspan(x[start], x[-1], alpha=0.3, color=color, linewidth=0, label=label)


def plot_sls_dls_on_rom(
    left_signal: np.ndarray,
    right_signal: np.ndarray,
    left_phases: dict[str, list[tuple[int, int]]],
    right_phases: dict[str, list[tuple[int, int]]],
    title: str | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot both hip ROM signals with SLS and DLS intervals shaded.

    Overlays three shading layers on the temporal ROM signals:

    - **Blue**  — Single Limb Support left  (left in stance, right in swing)
    - **Green** — Single Limb Support right (right in stance, left in swing)
    - **Red**   — Double Limb Support       (both legs simultaneously in stance)

    Args:
        left_signal: Preprocessed left hip ROM array.
        right_signal: Preprocessed right hip ROM array (same length as left).
        left_phases: Phase dict for the left leg (keys: ``'swing'``, ``'stance'``).
        right_phases: Phase dict for the right leg.
        title: Figure title.
        output_path: If given, save the figure here.

    Returns:
        Matplotlib Figure.
    """
    n = min(len(left_signal), len(right_signal))
    left_sig = left_signal[:n]
    right_sig = right_signal[:n]

    def _stance_mask(phases: dict, length: int) -> np.ndarray:
        mask = np.zeros(length, dtype=bool)
        for s, e in phases.get(STANCE, []):
            mask[min(s, length) : min(e, length)] = True
        return mask

    left_stance = _stance_mask(left_phases, n)
    right_stance = _stance_mask(right_phases, n)
    sls_left = left_stance & ~right_stance
    sls_right = right_stance & ~left_stance
    dls = left_stance & right_stance

    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    for ax, signal, side_label in zip(
        axes,
        [left_sig, right_sig],
        ["Left hip", "Right hip"],
        strict=True,
    ):
        ax.plot(x, signal, color="0.25", linewidth=0.9, zorder=4)
        _shade_mask(ax, x, sls_left, "#377eb8", "SLS — left leg only")
        _shade_mask(ax, x, sls_right, "#4daf4a", "SLS — right leg only")
        _shade_mask(ax, x, dls, "#e41a1c", "DLS — both legs")
        ax.set_ylabel(f"{side_label} (°)")
        ax.grid(True, alpha=0.25)

    axes[1].set_xlabel("Samples")

    patches = [
        mpatches.Patch(color="#377eb8", alpha=0.4, label="SLS — left leg only"),
        mpatches.Patch(color="#4daf4a", alpha=0.4, label="SLS — right leg only"),
        mpatches.Patch(color="#e41a1c", alpha=0.4, label="DLS — both legs"),
    ]
    axes[0].legend(handles=patches, fontsize=8, loc="upper right")
    axes[0].set_title(title or "SLS and DLS phases overlaid on hip ROM", fontsize=11)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
