"""Statistical visualization — replaces the R notebook (notebookTablasRJupyter.ipynb).

All functions receive a results DataFrame produced by
:func:`~gait_analysis.pipeline.run_pipeline` and generate figures replicating
the paper's style.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gait_analysis.config import VELOCITIES, VELOCITY_LABELS, WEIGHT_SUPPORT_LABELS, WEIGHT_SUPPORTS
from gait_analysis.schema import VELOCITY, WEIGHT_SUPPORT

# Derived directly from config so that adding a condition requires only one edit
_VEL_COLORS = ["#4daf4a", "#377eb8", "#e41a1c"]
_VELOCITY_PALETTE: dict[str, str] = dict(zip(VELOCITIES, _VEL_COLORS, strict=True))
_VEL_ORDER: list[str] = list(VELOCITIES)
_BWS_ORDER: list[int] = list(WEIGHT_SUPPORTS)


def _apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with velocity and BWS columns replaced by English display labels."""
    out = df.copy()
    if VELOCITY in out.columns:
        out[VELOCITY] = out[VELOCITY].map(VELOCITY_LABELS).fillna(out[VELOCITY])
    if WEIGHT_SUPPORT in out.columns:
        out[WEIGHT_SUPPORT] = (
            out[WEIGHT_SUPPORT].map(WEIGHT_SUPPORT_LABELS).fillna(out[WEIGHT_SUPPORT].astype(str))
        )
    return out


_VEL_ORDER_LABELS: list[str] = [VELOCITY_LABELS[v] for v in _VEL_ORDER]
_BWS_ORDER_LABELS: list[str] = [WEIGHT_SUPPORT_LABELS[b] for b in _BWS_ORDER]


def plot_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    group_by: str = WEIGHT_SUPPORT,
    hue: str = VELOCITY,
    ylabel: str | None = None,
    title: str | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Box plot of a metric grouped by condition (replicates paper Fig. 5 style).

    Args:
        df: Results DataFrame from the pipeline.
        metric: Column name of the metric to plot.
        group_by: Column to use for x-axis grouping (default: weight_support).
        hue: Column for color-coding (default: velocity).
        ylabel: Y-axis label (defaults to *metric*).
        title: Figure title.
        output_path: If given, save the figure here.

    Returns:
        Matplotlib Figure.
    """
    data = _apply_labels(df)
    x_order = _BWS_ORDER_LABELS if group_by == WEIGHT_SUPPORT else _VEL_ORDER_LABELS
    hue_order = _VEL_ORDER_LABELS if hue == VELOCITY else _BWS_ORDER_LABELS
    vel_palette_labels = {VELOCITY_LABELS[k]: v for k, v in _VELOCITY_PALETTE.items()}

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=data,
        x=group_by,
        y=metric,
        hue=hue,
        order=x_order,
        hue_order=hue_order,
        palette=vel_palette_labels if hue == VELOCITY else None,
        ax=ax,
        width=0.6,
        linewidth=0.8,
        fliersize=3,
    )
    ax.set_xlabel(group_by.replace("_", " ").title())
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    ax.set_title(title or metric.replace("_", " ").title(), fontsize=11)
    ax.legend(title=hue.replace("_", " ").title(), fontsize=8, title_fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_mad_bar_chart(
    df: pd.DataFrame,
    tibialis_col: str,
    gastrocnemius_col: str,
    group_by: str = VELOCITY,
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of MAD for tibialis anterior and gastrocnemius (paper Fig. 7 style).

    Args:
        df: Results DataFrame.
        tibialis_col: Column name for tibialis MAD.
        gastrocnemius_col: Column name for gastrocnemius MAD.
        group_by: Column for x-axis grouping (velocity or weight_support).
        output_path: If given, save the figure here.

    Returns:
        Matplotlib Figure.
    """
    data = _apply_labels(df)
    order = _VEL_ORDER_LABELS if group_by == VELOCITY else _BWS_ORDER_LABELS
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, col, title in zip(
        axes,
        [tibialis_col, gastrocnemius_col],
        ["Tibialis Anterior — MAD (%)", "Gastrocnemius — MAD (%)"],
        strict=False,
    ):
        means = data.groupby(group_by)[col].mean().reindex(order)
        sems = data.groupby(group_by)[col].sem().reindex(order)
        bars = ax.bar(
            range(len(order)),
            means,
            yerr=sems,
            color=list(_VELOCITY_PALETTE.values()) if group_by == VELOCITY else "#377eb8",
            capsize=4,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([str(v) for v in order])
        ax.set_xlabel(group_by.replace("_", " ").title())
        ax.set_ylabel("MAD (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        sns.despine(ax=ax)

        for bar, val in zip(bars, means, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Muscle Activation Duration by condition", fontsize=11, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_coactivation_scatter(
    df: pd.DataFrame,
    ci_col: str,
    x_col: str = WEIGHT_SUPPORT,
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of Coactivation Index vs a condition variable with a trend line.

    Args:
        df: Results DataFrame.
        ci_col: Column name for the CI metric.
        x_col: Column for x-axis (``weight_support`` or ``velocity``).
        output_path: If given, save the figure here.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=df,
        x=x_col,
        y=ci_col,
        scatter_kws={"alpha": 0.5, "s": 30},
        line_kws={"linewidth": 1.5},
        ax=ax,
        color="#377eb8",
    )
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Coactivation Index (%)")
    ax.set_title(f"Coactivation Index vs {x_col.replace('_', ' ')}", fontsize=11)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cadence_by_velocity(
    df: pd.DataFrame,
    cadence_col: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Boxplot of cadence grouped by walking speed, colored by BWS level.

    Args:
        df: Results DataFrame from the pipeline.
        cadence_col: Column name for cadence (steps/min).
        output_path: If given, save the figure here.

    Returns:
        Matplotlib Figure.
    """
    data = _apply_labels(df.dropna(subset=[cadence_col]))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=data,
        x=VELOCITY,
        y=cadence_col,
        hue=WEIGHT_SUPPORT,
        order=_VEL_ORDER_LABELS,
        palette="Blues_d",
        ax=ax,
        width=0.6,
        linewidth=0.8,
        fliersize=3,
    )
    ax.set_xlabel("Walking speed")
    ax.set_ylabel("Cadence (steps/min)")
    ax.set_title("Cadence by walking speed and body weight support", fontsize=11)
    ax.legend(title="BWS (%)", fontsize=8, title_fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sls_dls(
    df: pd.DataFrame,
    sls_col: str,
    dls_col: str,
    group_by: str = VELOCITY,
    output_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing SLS and DLS percentages across conditions.

    Single Limb Support (SLS) and Double Limb Support (DLS) are shown
    side-by-side per condition, illustrating how gait speed and body weight
    support affect stance-phase overlap between legs.

    Args:
        df: Results DataFrame from the pipeline.
        sls_col: Column name for SLS percentage (one side).
        dls_col: Column name for DLS percentage.
        group_by: Column for x-axis grouping (velocity or weight_support).
        output_path: If given, save the figure here.

    Returns:
        Matplotlib Figure.
    """
    order = _VEL_ORDER_LABELS if group_by == VELOCITY else _BWS_ORDER_LABELS
    data = _apply_labels(df.dropna(subset=[sls_col, dls_col]))

    melted = data[[group_by, sls_col, dls_col]].melt(
        id_vars=group_by,
        value_vars=[sls_col, dls_col],
        var_name="Phase",
        value_name="Percentage (%)",
    )
    melted["Phase"] = melted["Phase"].map({sls_col: "SLS", dls_col: "DLS"})

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=melted,
        x=group_by,
        y="Percentage (%)",
        hue="Phase",
        order=order,
        palette={"SLS": "#377eb8", "DLS": "#e41a1c"},
        ax=ax,
        width=0.6,
        capsize=0.05,
        err_kws={"linewidth": 1.2},
    )
    ax.set_xlabel(group_by.replace("_", " ").title())
    ax.set_ylabel("Time in phase (%)")
    ax.set_title("Single and double limb support by condition", fontsize=11)
    ax.legend(title="Phase", fontsize=9, title_fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
