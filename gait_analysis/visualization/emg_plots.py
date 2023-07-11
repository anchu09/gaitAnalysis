from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_emg_raw(
    signals: dict[str, np.ndarray],
    title: str = "Raw EMG signals",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot raw EMG signals for all four muscles.

    Args:
        signals: Dict mapping muscle name → 1-D amplitude array.
        title: Figure title.
        output_path: If given, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for ax, (name, signal), color in zip(axes, signals.items(), colors, strict=False):
        ax.plot(signal, color=color, linewidth=0.8)
        ax.set_ylabel(name.replace("_", " ").title(), fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Samples")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_binary_contractions(
    signal: np.ndarray,
    contractions: np.ndarray,
    muscle_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Overlay binary contraction mask on normalized EMG signal.

    Args:
        signal: 1-D EMG amplitude array (normalized to [0, 1]).
        contractions: 1-D binary array (0/1).
        muscle_name: Label for the y-axis.
        output_path: If given, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(signal, color="#377eb8", linewidth=0.8, label="EMG (normalized)")
    ax.fill_between(
        range(len(contractions)),
        contractions.astype(float),
        alpha=0.3,
        color="#e41a1c",
        label="Active",
    )
    ax.set_xlabel("Samples")
    ax.set_ylabel(muscle_name.replace("_", " ").title())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_emg_coactivation(
    agonist: np.ndarray,
    antagonist: np.ndarray,
    agonist_label: str = "Tibialis anterior",
    antagonist_label: str = "Gastrocnemius",
    title: str | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot two EMG envelopes with the shared (coactivation) area highlighted.

    The filled area between both curves where both muscles are simultaneously
    active visualizes the contribution to the Coactivation Index (CI).

    Args:
        agonist: 1-D EMG envelope for the agonist muscle (e.g. tibialis anterior).
        antagonist: 1-D EMG envelope for the antagonist (e.g. gastrocnemius).
        agonist_label: Legend label for the agonist.
        antagonist_label: Legend label for the antagonist.
        title: Figure title.
        output_path: If given, save the figure here.

    Returns:
        Matplotlib Figure.
    """

    # Normalize both signals to [0, 1] for visual comparison
    def _norm(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    ag = _norm(agonist)
    ant = _norm(antagonist)
    x = np.arange(len(ag))

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(x, ag, color="#377eb8", linewidth=1.2, label=agonist_label)
    ax.plot(x, ant, color="#e41a1c", linewidth=1.2, label=antagonist_label)

    # Shared area = coactivation
    ax.fill_between(x, np.minimum(ag, ant), alpha=0.35, color="#984ea3", label="Coactivation area")

    ax.set_xlabel("Samples")
    ax.set_ylabel("Normalized EMG amplitude")
    ax.set_title(title or f"EMG coactivation — {agonist_label} vs {antagonist_label}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
