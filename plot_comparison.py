"""
plot_comparison.py — Visualize model_comparison.csv results.

Usage:
    python plot_comparison.py
    python plot_comparison.py --input outputs/tables/model_comparison.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

FIGURE_DIR = Path("outputs/figures")
ARCH_ORDER = ["IPCA", "CA0", "CA1", "CA2", "CA3"]
ARCH_COLORS = {
    "IPCA": "#555555",
    "CA0":  "#2196F3",
    "CA1":  "#4CAF50",
    "CA2":  "#FF9800",
    "CA3":  "#E91E63",
}
MARKERS = {"IPCA": "s", "CA0": "o", "CA1": "^", "CA2": "D", "CA3": "v"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["architecture"] = pd.Categorical(df["architecture"], categories=ARCH_ORDER, ordered=True)
    return df.sort_values(["architecture", "K"]).reset_index(drop=True)


# ── Figure 1: R²_total by K, one line per architecture ──────────────────────

def plot_r2_total(df: pd.DataFrame, ax: plt.Axes):
    for arch in ARCH_ORDER:
        sub = df[df["architecture"] == arch]
        ax.plot(sub["K"], sub["r2_total"] * 100,
                color=ARCH_COLORS[arch], marker=MARKERS[arch],
                linewidth=1.8, markersize=6, label=arch)
    ax.set_xlabel("Number of factors K")
    ax.set_ylabel("Total R² (%)")
    ax.set_title("Out-of-Sample Total R²")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(title="Architecture", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)


# ── Figure 2: EW Sharpe ratio by K ──────────────────────────────────────────

def plot_sharpe_ew(df: pd.DataFrame, ax: plt.Axes):
    for arch in ARCH_ORDER:
        sub = df[df["architecture"] == arch]
        ax.plot(sub["K"], sub["sharpe_ew"],
                color=ARCH_COLORS[arch], marker=MARKERS[arch],
                linewidth=1.8, markersize=6, label=arch)
    ax.set_xlabel("Number of factors K")
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title("EW Long-Only Sharpe Ratio")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(title="Architecture", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)


# ── Figure 3: Heatmap — R²_total across arch × K ────────────────────────────

def plot_heatmap(df: pd.DataFrame, ax: plt.Axes, metric: str, title: str, fmt: str = ".2f"):
    pivot = df.pivot(index="architecture", columns="K", values=metric)
    pivot = pivot.reindex(ARCH_ORDER)

    scale = 100 if "r2" in metric else 1
    data = pivot.values * scale

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                 label="%" if "r2" in metric else "")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("K")
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:{fmt}}", ha="center", va="center",
                    fontsize=8, color="black")


# ── Figure 4: Bar chart — best K per architecture ───────────────────────────

def plot_best_k_bars(df: pd.DataFrame, ax: plt.Axes):
    best = df.loc[df.groupby("architecture")["r2_total"].idxmax()].copy()
    best = best.sort_values("architecture")

    bars = ax.bar(
        best["architecture"],
        best["r2_total"] * 100,
        color=[ARCH_COLORS[a] for a in best["architecture"]],
        edgecolor="white", linewidth=0.8,
    )
    for bar, row in zip(bars, best.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"K={row.K}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Total R² (%)")
    ax.set_title("Best R²_total per Architecture (optimal K)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, best["r2_total"].max() * 100 * 1.15)


# ── Figure 5: Scatter R²_total vs Sharpe (EW), annotated by arch+K ──────────

def plot_scatter(df: pd.DataFrame, ax: plt.Axes):
    for arch in ARCH_ORDER:
        sub = df[df["architecture"] == arch]
        ax.scatter(sub["r2_total"] * 100, sub["sharpe_ew"],
                   color=ARCH_COLORS[arch], marker=MARKERS[arch],
                   s=60, label=arch, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(f"K{int(row['K'])}",
                        (row["r2_total"] * 100, row["sharpe_ew"]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=6.5, color=ARCH_COLORS[arch])
    ax.set_xlabel("Total R² (%)")
    ax.set_ylabel("EW Sharpe ratio")
    ax.set_title("R²_total vs Sharpe (EW)")
    ax.legend(title="Architecture", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/tables/model_comparison.csv")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = load(args.input)

    # ── Combined summary figure (2×2 + wide scatter) ──
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1:])

    plot_r2_total(df, ax1)
    plot_sharpe_ew(df, ax2)
    plot_heatmap(df, ax3, "r2_total", "R²_total Heatmap (%)")
    plot_best_k_bars(df, ax4)
    plot_scatter(df, ax5)

    fig.suptitle("Autoencoder Asset Pricing — Model Comparison", fontsize=13, fontweight="bold", y=1.01)
    out = FIGURE_DIR / "model_comparison.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")

    out_png = FIGURE_DIR / "model_comparison.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_png}")
    plt.close(fig)

    # ── Standalone heatmaps for Sharpe ──
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_heatmap(df, axes[0], "r2_total", "R²_total (%)", fmt=".2f")
    plot_heatmap(df, axes[1], "sharpe_ew", "EW Sharpe", fmt=".2f")
    fig2.suptitle("Architecture × K Heatmaps", fontsize=11, fontweight="bold")
    fig2.tight_layout()
    out2 = FIGURE_DIR / "heatmaps.png"
    fig2.savefig(out2, bbox_inches="tight", dpi=150)
    print(f"Saved: {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
