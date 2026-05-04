"""
plot_all.py — Visualize all tables in outputs/tables/.

Usage:
    python plot_all.py
    python plot_all.py --tables outputs/tables --figures outputs/figures
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Shared style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

ARCH_ORDER  = ["IPCA", "CA0", "CA1", "CA2", "CA3"]
ARCH_COLORS = {
    "IPCA": "#555555",
    "CA0":  "#2196F3",
    "CA1":  "#4CAF50",
    "CA2":  "#FF9800",
    "CA3":  "#E91E63",
}
MARKERS = {"IPCA": "s", "CA0": "o", "CA1": "^", "CA2": "D", "CA3": "v"}

STRAT_COLORS = {
    "Long-Short (AE)":  "#1f77b4",
    "Long Only":        "#2ca02c",
    "Chars-Only L/S":   "#ff7f0e",
    "Short Only":       "#9467bd",
    "Long-Short":       "#1f77b4",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, figs: Path, stem: str):
    for ext in ("pdf", "png"):
        p = figs / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
    print(f"Saved: {stem}")


def arch_heatmap(df: pd.DataFrame, ax: plt.Axes, metric: str, title: str,
                 fmt: str = ".2f"):
    pivot = df.pivot(index="architecture", columns="K", values=metric).reindex(ARCH_ORDER)
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
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=8, color="black")


def _pct_fmt(x, _):
    return f"{x:.0f}%"


# ── model_comparison.csv ──────────────────────────────────────────────────────

def plot_model_comparison(tables: Path, figs: Path):
    df = pd.read_csv(tables / "model_comparison.csv")
    df["architecture"] = pd.Categorical(df["architecture"], categories=ARCH_ORDER, ordered=True)
    df = df.sort_values(["architecture", "K"]).reset_index(drop=True)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.36)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1:])

    # R² by K
    for arch in ARCH_ORDER:
        sub = df[df["architecture"] == arch]
        ax1.plot(sub["K"], sub["r2_total"] * 100,
                 color=ARCH_COLORS[arch], marker=MARKERS[arch],
                 linewidth=1.8, markersize=6, label=arch)
    ax1.set_xlabel("K"); ax1.set_ylabel("Total R² (%)")
    ax1.set_title("Out-of-Sample Total R²")
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax1.legend(title="Architecture")
    ax1.grid(True)

    # Predictive Sharpe by K
    for arch in ARCH_ORDER:
        sub = df[df["architecture"] == arch]
        ax2.plot(sub["K"], sub["sharpe_pred"],
                 color=ARCH_COLORS[arch], marker=MARKERS[arch],
                 linewidth=1.8, markersize=6, label=arch)
    ax2.set_xlabel("K"); ax2.set_ylabel("Annualized Sharpe")
    ax2.set_title("Predictive Sharpe Ratio")
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax2.legend(title="Architecture")
    ax2.grid(True)

    # R² heatmap
    arch_heatmap(df, ax3, "r2_total", "R²_total Heatmap (%)")

    # Best K bars
    best = df.loc[df.groupby("architecture")["r2_total"].idxmax()].sort_values("architecture")
    bars = ax4.bar(best["architecture"], best["r2_total"] * 100,
                   color=[ARCH_COLORS[a] for a in best["architecture"]],
                   edgecolor="white", linewidth=0.8)
    for bar, row in zip(bars, best.itertuples()):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"K={row.K}", ha="center", va="bottom", fontsize=8.5)
    ax4.set_ylabel("Total R² (%)"); ax4.set_title("Best R²_total per Architecture")
    ax4.grid(True, axis="y")
    ax4.set_ylim(0, best["r2_total"].max() * 100 * 1.18)

    # Scatter R² vs Sharpe
    for arch in ARCH_ORDER:
        sub = df[df["architecture"] == arch]
        ax5.scatter(sub["r2_total"] * 100, sub["sharpe_pred"],
                    color=ARCH_COLORS[arch], marker=MARKERS[arch],
                    s=60, label=arch, zorder=3)
        for _, row in sub.iterrows():
            ax5.annotate(f"K{int(row['K'])}", (row["r2_total"] * 100, row["sharpe_pred"]),
                         textcoords="offset points", xytext=(4, 3),
                         fontsize=6.5, color=ARCH_COLORS[arch])
    ax5.set_xlabel("Total R² (%)"); ax5.set_ylabel("Predictive Sharpe")
    ax5.set_title("R²_total vs Predictive Sharpe")
    ax5.legend(title="Architecture")
    ax5.grid(True)

    fig.suptitle("Autoencoder Asset Pricing — Model Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    savefig(fig, figs, "model_comparison")
    plt.close(fig)

    # Standalone heatmaps
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
    arch_heatmap(df, axes[0], "r2_total",   "R²_total (%)",       fmt=".2f")
    arch_heatmap(df, axes[1], "sharpe_pred", "Predictive Sharpe", fmt=".2f")
    fig2.suptitle("Architecture × K Heatmaps", fontsize=11, fontweight="bold")
    fig2.tight_layout()
    savefig(fig2, figs, "heatmaps")
    plt.close(fig2)


# ── plan_a_portfolio_sorts.csv ────────────────────────────────────────────────

def plot_plan_a_portfolio_sorts(tables: Path, figs: Path):
    sorts = pd.read_csv(tables / "plan_a_portfolio_sorts.csv")
    quintiles = sorts[sorts["quintile"] != 0].copy()
    spread = sorts[sorts["quintile"] == 0].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    colors_q = ["#d62728" if i == len(quintiles) - 1
                else "#1f77b4" if i == 0
                else "#aec7e8"
                for i in range(len(quintiles))]

    ax = axes[0]
    bars = ax.bar(quintiles["label"], quintiles["mean_return_monthly"] * 100,
                  color=colors_q, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Mean Monthly Return by Quintile")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Anomaly Quintile")
    ax.tick_params(axis="x", rotation=20)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (0.01 if h >= 0 else -0.03),
                f"{h:.2f}%", ha="center", va="bottom" if h >= 0 else "top",
                fontsize=8)

    ax = axes[1]
    bars = ax.bar(quintiles["label"], quintiles["sharpe_annual"],
                  color=colors_q, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Annualized Sharpe Ratio by Quintile")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Anomaly Quintile")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (0.02 if h >= 0 else -0.05),
                f"{h:.2f}", ha="center", va="bottom" if h >= 0 else "top",
                fontsize=8)

    fig.suptitle(
        f"Plan A — Portfolio Sorts  |  High−Low spread: "
        f"{spread['mean_return_monthly']*100:.3f}%/mo, Sharpe {spread['sharpe_annual']:.2f}",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, figs, "plan_a_portfolio_sorts")
    plt.close(fig)


# ── plan_a_transitions.csv ────────────────────────────────────────────────────

def plot_plan_a_transitions(tables: Path, figs: Path):
    trans = pd.read_csv(tables / "plan_a_transitions.csv")
    groups = trans[trans["group"] != "Transition − Control (spread)"].copy()
    spread_row = trans[trans["group"] == "Transition − Control (spread)"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bar_colors = ["#d62728", "#1f77b4"]
    bars = ax.bar(["Low→High\n(Transition)", "Low→Low\n(Control)"],
                  groups["mean_cum_ret"].values * 100,
                  color=bar_colors, edgecolor="white", width=0.45)
    ax.errorbar(["Low→High\n(Transition)", "Low→Low\n(Control)"],
                groups["mean_cum_ret"].values * 100,
                yerr=groups["std"].values * 100,
                fmt="none", color="black", capsize=5, linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Plan A — Transition Analysis")
    ax.set_ylabel("Mean Cumulative Return (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(True, axis="y")
    ax.annotate(
        f"Spread: {spread_row['mean_cum_ret']*100:.2f}%  (Sharpe {spread_row['sharpe_annual']:.2f})",
        xy=(0.5, 0.04), xycoords="axes fraction", ha="center", fontsize=9,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    savefig(fig, figs, "plan_a_transitions")
    plt.close(fig)


# ── plan_a_hl_significance_*.csv ──────────────────────────────────────────────

def plot_plan_a_hl_significance(tables: Path, figs: Path):
    anom = pd.read_csv(tables / "plan_a_hl_significance_anomaly.csv")
    pred = pd.read_csv(tables / "plan_a_hl_significance_pred.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, df, title in [
        (axes[0], anom, "Anomaly Score H−L Significance"),
        (axes[1], pred, "Predicted Return H−L Significance"),
    ]:
        colors = ["#2ecc71" if sig else "#e74c3c" for sig in df["significant_5pct"]]
        bars = ax.barh(df["test"], df["t_stat"], color=colors, edgecolor="white", height=0.55)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline( 1.96, color="#888888", linewidth=1.0, linestyle="--", label="±1.96 (5%)")
        ax.axvline(-1.96, color="#888888", linewidth=1.0, linestyle="--")
        ax.set_xlabel("t-statistic")
        ax.set_title(title)

        # color-coded significance indicator in bar labels
        xlim_max = max(abs(df["t_stat"].max()), abs(df["t_stat"].min()), 2.5) * 1.35
        ax.set_xlim(-xlim_max, xlim_max)
        for bar, row in zip(bars, df.itertuples()):
            label = f"t={row.t_stat:.2f}  p={row.p_value:.3f}"
            x = bar.get_width() + 0.1 if bar.get_width() >= 0 else bar.get_width() - 0.1
            ha = "left" if bar.get_width() >= 0 else "right"
            ax.text(x, bar.get_y() + bar.get_height() / 2, label,
                    va="center", fontsize=8, color="black")
        ax.legend(fontsize=8)

    # legend for significance color
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#2ecc71", label="Significant (5%)"),
                  Patch(facecolor="#e74c3c", label="Not significant")]
    fig.legend(handles=legend_els, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Plan A — High−Low Spread Significance Tests", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, figs, "plan_a_hl_significance")
    plt.close(fig)


# ── plan_b_ablation.csv ───────────────────────────────────────────────────────

def plot_plan_b_ablation(tables: Path, figs: Path):
    abl = pd.read_csv(tables / "plan_b_ablation.csv")
    lambdas = sorted(abl["energy_lambda"].unique())
    gammas  = sorted(abl["disentangle_gamma"].unique())

    def make_heatmap(pivot, title, fmt, ax, cmap="viridis"):
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xticks(range(len(gammas)))
        ax.set_xticklabels([str(g) for g in gammas], fontsize=8)
        ax.set_yticks(range(len(lambdas)))
        ax.set_yticklabels([str(l) for l in lambdas], fontsize=8)
        ax.set_xlabel("disentangle_gamma")
        ax.set_ylabel("energy_lambda")
        ax.set_title(title)
        mean_val = pivot.values.mean()
        for i in range(len(lambdas)):
            for j in range(len(gammas)):
                v = pivot.values[i, j]
                ax.text(j, i, fmt % v, ha="center", va="center",
                        fontsize=7, color="white" if v < mean_val else "black")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for metric, title, fmt, ax in [
        ("r2_total",       "R² Total",      "%.3f", axes[0]),
        ("r2_pred",        "R² Predictive", "%.3f", axes[1]),
        ("final_val_loss", "Val Loss",      "%.4f", axes[2]),
    ]:
        pivot = (abl.pivot(index="energy_lambda", columns="disentangle_gamma", values=metric)
                    .reindex(index=lambdas, columns=gammas))
        cmap = "viridis_r" if metric == "final_val_loss" else "viridis"
        make_heatmap(pivot, title, fmt, ax, cmap=cmap)

    fig.suptitle("Plan B — Ablation: energy_lambda × disentangle_gamma", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, figs, "plan_b_ablation")
    plt.close(fig)


# ── backtest tables ───────────────────────────────────────────────────────────

def plot_backtest(tables: Path, figs: Path):
    perf   = pd.read_csv(tables / "backtest_performance.csv")
    rets   = pd.read_csv(tables / "backtest_monthly_returns.csv")
    roll   = pd.read_csv(tables / "backtest_rolling_sharpe.csv")
    alphas = pd.read_csv(tables / "backtest_alphas.csv")
    dd     = pd.read_csv(tables / "backtest_drawdown.csv")

    rets["date"] = pd.to_datetime(rets["date"].astype(str), format="%Y%m")
    roll["date"] = pd.to_datetime(roll["date"].astype(str), format="%Y%m")
    dd["date"]   = pd.to_datetime(dd["date"].astype(str).str.split(".").str[0], format="%Y%m")

    rets_sorted = rets.sort_values("date").copy()
    # growth of $1 (log scale makes more sense for compounding)
    rets_sorted["ls_wealth"]  = (1 + rets_sorted["ls_ret"]).cumprod()
    rets_sorted["lo_wealth"]  = (1 + rets_sorted["long_ret"]).cumprod()
    rets_sorted["mkt_wealth"] = (1 + rets_sorted["market_ret"]).cumprod()
    # also keep chars-only if present
    if "chars_ls_ret" in rets_sorted.columns:
        rets_sorted["chars_wealth"] = (1 + rets_sorted["chars_ls_ret"]).cumprod()

    # ── Figure 1: Growth of $1 (log scale) + drawdown ──
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.4], hspace=0.08)
    ax_ret = fig.add_subplot(gs[0])
    ax_dd  = fig.add_subplot(gs[1], sharex=ax_ret)

    # top: growth of $1, log-y
    ax_ret.plot(rets_sorted["date"], rets_sorted["ls_wealth"],
                label="Long-Short (AE)", linewidth=2.0, color="#1f77b4")
    ax_ret.plot(rets_sorted["date"], rets_sorted["lo_wealth"],
                label="Long Only", linewidth=2.0, color="#2ca02c")
    if "chars_wealth" in rets_sorted.columns:
        ax_ret.plot(rets_sorted["date"], rets_sorted["chars_wealth"],
                    label="Chars-Only L/S", linewidth=1.5, color="#ff7f0e", linestyle="--")
    ax_ret.plot(rets_sorted["date"], rets_sorted["mkt_wealth"],
                label="Market", linewidth=1.4, linestyle=":", color="#888888")
    ax_ret.axhline(1, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:.0f}"))
    ax_ret.set_ylabel("Growth of $1  (log scale)")
    ax_ret.set_title("Cumulative Wealth — Growth of $1 (Log Scale)")
    ax_ret.legend(loc="upper left")
    ax_ret.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.setp(ax_ret.get_xticklabels(), visible=False)

    # annotation: final values
    for col, label, color in [
        ("ls_wealth",    "L/S AE",  "#1f77b4"),
        ("lo_wealth",    "L/O",     "#2ca02c"),
        ("mkt_wealth",   "Market",  "#888888"),
    ]:
        last = rets_sorted[col].iloc[-1]
        ann_ret = perf.loc[perf["strategy"].str.contains(
            "Long-Short" if "ls" in col else ("Long Only" if "lo" in col else "Market"),
            case=False, na=False), "ann_return"]
        ann_str = f"  {ann_ret.values[0]*100:.1f}%/yr" if len(ann_ret) else ""
        ax_ret.annotate(f"${last:.0f}{ann_str}",
                        xy=(rets_sorted["date"].iloc[-1], last),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=color, va="center",
                        annotation_clip=False)

    # bottom: drawdown
    for strat, color, alpha in [
        ("Long-Short", "#E91E63", 0.55),
        ("Long Only",  "#2196F3", 0.30),
    ]:
        sdd = dd[dd["strategy"] == strat].sort_values("date")
        ax_dd.fill_between(sdd["date"], sdd["drawdown"] * 100, 0,
                           alpha=alpha, color=color, label=f"{strat}")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.set_xlabel("")
    ax_dd.legend(fontsize=8, loc="lower left")
    ax_dd.grid(True)

    fig.suptitle("Backtest — Cumulative Returns & Drawdown",
                 fontsize=12, fontweight="bold")
    savefig(fig, figs, "backtest_returns_drawdown")
    plt.close(fig)

    # ── Figure 2: Rolling Sharpe + monthly return distribution ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), gridspec_kw={"hspace": 0.42})

    ax = axes[0]
    roll_sorted = roll.sort_values("date")
    ax.plot(roll_sorted["date"], roll_sorted["rolling_sharpe_36m"],
            color="#FF9800", linewidth=1.8, label="36-mo Rolling Sharpe")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="#4CAF50", linewidth=0.8, linestyle="--", alpha=0.7, label="SR = 1")
    ax.axhline(2, color="#1f77b4", linewidth=0.8, linestyle="--", alpha=0.7, label="SR = 2")
    ax.fill_between(roll_sorted["date"], roll_sorted["rolling_sharpe_36m"], 0,
                    where=roll_sorted["rolling_sharpe_36m"] >= 0,
                    alpha=0.15, color="#FF9800")
    ax.fill_between(roll_sorted["date"], roll_sorted["rolling_sharpe_36m"], 0,
                    where=roll_sorted["rolling_sharpe_36m"] < 0,
                    alpha=0.15, color="#E91E63")
    ax.set_ylabel("Rolling Sharpe Ratio")
    ax.set_title("36-Month Rolling Sharpe — Long-Short Strategy")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)

    ax = axes[1]
    bar_colors = np.where(rets_sorted["ls_ret"] >= 0, "#4CAF50", "#E91E63")
    ax.bar(rets_sorted["date"], rets_sorted["ls_ret"] * 100,
           color=bar_colors, width=25, label="L/S monthly")
    ax.set_ylabel("Monthly Return (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Long-Short Monthly Returns")
    ax.axhline(0, color="black", linewidth=0.8)
    pos_pct = (rets_sorted["ls_ret"] >= 0).mean() * 100
    ax.text(0.02, 0.94, f"Hit rate: {pos_pct:.1f}%", transform=ax.transAxes,
            fontsize=8.5, va="top", color="#555555",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.7))
    ax.grid(True)

    fig.suptitle("Backtest — Rolling Sharpe & Monthly Returns",
                 fontsize=12, fontweight="bold")
    savefig(fig, figs, "backtest_rolling_monthly")
    plt.close(fig)

    # ── Figure 3: Performance summary (separate axes) + alpha table ──
    fig = plt.figure(figsize=(14, 5.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.38, width_ratios=[1, 1, 1.4])
    ax_ret_bar = fig.add_subplot(gs[0])
    ax_risk    = fig.add_subplot(gs[1])
    ax_tbl     = fig.add_subplot(gs[2])

    strats_ordered = perf["strategy"].tolist()
    colors_strat = [STRAT_COLORS.get(s, "#888888") for s in strats_ordered]
    x = np.arange(len(strats_ordered))
    width = 0.38

    # return + vol grouped bars
    ax_ret_bar.bar(x - width / 2, perf["ann_return"] * 100, width,
                   color=colors_strat, alpha=0.9, label="Ann. Return", edgecolor="white")
    ax_ret_bar.bar(x + width / 2, perf["ann_vol"] * 100, width,
                   color=colors_strat, alpha=0.45, label="Ann. Vol", edgecolor="white",
                   hatch="//")
    ax_ret_bar.set_xticks(x)
    ax_ret_bar.set_xticklabels(strats_ordered, rotation=22, ha="right", fontsize=8)
    ax_ret_bar.axhline(0, color="black", linewidth=0.8)
    ax_ret_bar.set_ylabel("(%)")
    ax_ret_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_ret_bar.set_title("Return & Volatility")
    ax_ret_bar.legend(fontsize=8)
    ax_ret_bar.grid(True, axis="y")

    # Sharpe + Max DD
    ax_risk.bar(x - width / 2, perf["sharpe"], width,
                color=colors_strat, alpha=0.9, label="Sharpe", edgecolor="white")
    ax_risk.bar(x + width / 2, perf["max_drawdown"] * 100, width,
                color=colors_strat, alpha=0.45, label="Max DD (%)", edgecolor="white",
                hatch="//")
    ax_risk.set_xticks(x)
    ax_risk.set_xticklabels(strats_ordered, rotation=22, ha="right", fontsize=8)
    ax_risk.axhline(0, color="black", linewidth=0.8)
    ax_risk.set_title("Sharpe & Max Drawdown")
    ax_risk.legend(fontsize=8)
    ax_risk.grid(True, axis="y")

    # alpha table
    ax_tbl.axis("off")
    col_labels = ["Model", "α/mo", "α/yr", "t-stat", "p-val", "β", "R²"]
    rows = []
    for _, r in alphas.iterrows():
        rows.append([
            r["model"],
            f"{r['alpha_monthly']:.4f}",
            f"{r['alpha_annual']:.3f}",
            f"{r['alpha_tstat']:.2f}",
            f"{r['alpha_pval']:.2e}",
            f"{r['market_beta']:.3f}",
            f"{r['r_squared']:.3f}",
        ])
    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 1.8)
    ax_tbl.set_title("Alpha Decomposition", pad=16)

    fig.suptitle("Backtest — Performance Summary & Alpha", fontsize=12, fontweight="bold")
    savefig(fig, figs, "backtest_performance_alpha")
    plt.close(fig)

    # ── Figure 4: Holdings ──
    long_h  = pd.read_csv(tables / "backtest_long_holdings.csv")
    short_h = pd.read_csv(tables / "backtest_short_holdings.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, df, side, color in [
        (axes[0, 0], long_h,  "Long",  "#4C72B0"),
        (axes[0, 1], short_h, "Short", "#C44E52"),
    ]:
        ax.hist(df["n_months"], bins=30, color=color, edgecolor="white", alpha=0.85)
        median_m = df["n_months"].median()
        ax.axvline(median_m, color="black", linestyle="--", linewidth=1.2,
                   label=f"Median = {median_m:.0f} mo")
        ax.set_xlabel("Months held"); ax.set_ylabel("# stocks")
        ax.set_title(f"{side} Holdings — Tenure Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y")

    for ax, df, side, color in [
        (axes[1, 0], long_h,  "Long",  "#4C72B0"),
        (axes[1, 1], short_h, "Short", "#C44E52"),
    ]:
        ax.scatter(df["mean_pred_ret"] * 100, df["mean_actual_ret"] * 100,
                   alpha=0.35, s=12, color=color)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y = x")
        ax.set_xlabel("Mean predicted return (%)")
        ax.set_ylabel("Mean actual return (%)")
        ax.set_title(f"{side} Holdings — Predicted vs Actual Return")
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle("Backtest — Holdings Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, figs, "backtest_holdings")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables",  default="outputs/tables")
    parser.add_argument("--figures", default="outputs/figures")
    args = parser.parse_args()

    tables = Path(args.tables)
    figs   = Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)

    plot_model_comparison(tables, figs)
    plot_plan_a_portfolio_sorts(tables, figs)
    plot_plan_a_transitions(tables, figs)
    plot_plan_a_hl_significance(tables, figs)
    plot_plan_b_ablation(tables, figs)
    plot_backtest(tables, figs)

    print(f"\nAll figures written to {figs}/")


if __name__ == "__main__":
    main()
