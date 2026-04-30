import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

TABLES = "outputs/tables"
FIGS = "outputs/figures"
os.makedirs(FIGS, exist_ok=True)


# ── Plan A: Portfolio Sorts ────────────────────────────────────────────────────

sorts = pd.read_csv(f"{TABLES}/plan_a_portfolio_sorts.csv")
quintiles = sorts[sorts["quintile"] != 0].copy()
spread = sorts[sorts["quintile"] == 0].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Mean monthly return by quintile
ax = axes[0]
colors = ["#4C72B0"] * 5
ax.bar(quintiles["label"], quintiles["mean_return_monthly"] * 100, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Mean Monthly Return by Quintile")
ax.set_ylabel("Return (%)")
ax.set_xlabel("Anomaly Quintile")
ax.tick_params(axis="x", rotation=20)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# Annualized Sharpe by quintile
ax = axes[1]
ax.bar(quintiles["label"], quintiles["sharpe_annual"], color="#DD8452", edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Annualized Sharpe Ratio by Quintile")
ax.set_ylabel("Sharpe Ratio")
ax.set_xlabel("Anomaly Quintile")
ax.tick_params(axis="x", rotation=20)

fig.suptitle(f"Plan A — Portfolio Sorts  |  High−Low spread: {spread['mean_return_monthly']*100:.3f}%/mo, Sharpe {spread['sharpe_annual']:.2f}", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIGS}/plan_a_portfolio_sorts.pdf", bbox_inches="tight")
fig.savefig(f"{FIGS}/plan_a_portfolio_sorts.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved plan_a_portfolio_sorts")


# ── Plan A: Transitions ────────────────────────────────────────────────────────

trans = pd.read_csv(f"{TABLES}/plan_a_transitions.csv")

fig, ax = plt.subplots(figsize=(6, 4))
groups = trans[trans["group"] != "Transition − Control (spread)"].copy()
spread_row = trans[trans["group"] == "Transition − Control (spread)"].iloc[0]

colors = ["#C44E52", "#4C72B0"]
bars = ax.bar(["Low→High\n(Transition)", "Low→Low\n(Control)"],
              groups["mean_cum_ret"].values * 100,
              color=colors, edgecolor="white", width=0.5)

# error bars (1 std)
ax.errorbar(["Low→High\n(Transition)", "Low→Low\n(Control)"],
            groups["mean_cum_ret"].values * 100,
            yerr=groups["std"].values * 100,
            fmt="none", color="black", capsize=4, linewidth=1)

ax.set_title("Plan A — Transition Analysis\nMean Cumulative Return (±1 SD)")
ax.set_ylabel("Cumulative Return (%)")
ax.annotate(f"Spread: {spread_row['mean_cum_ret']*100:.2f}%  (Sharpe {spread_row['sharpe_annual']:.2f})",
            xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(f"{FIGS}/plan_a_transitions.pdf", bbox_inches="tight")
fig.savefig(f"{FIGS}/plan_a_transitions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved plan_a_transitions")


# ── Plan B: Ablation heatmaps ──────────────────────────────────────────────────

abl = pd.read_csv(f"{TABLES}/plan_b_ablation.csv")
lambdas = sorted(abl["energy_lambda"].unique())
gammas = sorted(abl["disentangle_gamma"].unique())

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
    for i in range(len(lambdas)):
        for j in range(len(gammas)):
            ax.text(j, i, fmt % pivot.values[i, j], ha="center", va="center",
                    fontsize=7, color="white" if pivot.values[i, j] < pivot.values.mean() else "black")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for metric, title, fmt, ax in [
    ("r2_total",    "R² Total",         "%.3f", axes[0]),
    ("r2_pred",     "R² Predictive",    "%.3f", axes[1]),
    ("final_val_loss", "Val Loss",      "%.4f", axes[2]),
]:
    pivot = abl.pivot(index="energy_lambda", columns="disentangle_gamma", values=metric)
    pivot = pivot.reindex(index=lambdas, columns=gammas)
    cmap = "viridis" if metric != "final_val_loss" else "viridis_r"
    make_heatmap(pivot, title, fmt, ax, cmap=cmap)

fig.suptitle("Plan B — Ablation: energy_lambda × disentangle_gamma", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIGS}/plan_b_ablation.pdf", bbox_inches="tight")
fig.savefig(f"{FIGS}/plan_b_ablation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved plan_b_ablation")


# ── Model Comparison ───────────────────────────────────────────────────────────

comp = pd.read_csv(f"{TABLES}/model_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

arch_order = ["IPCA", "CA0", "CA1", "CA2", "CA3"]
markers = ["o", "s", "^", "D", "v"]
colors_map = dict(zip(arch_order, plt.cm.tab10.colors))

for metric, ylabel, ax in [
    ("r2_total", "R² Total", axes[0]),
    ("sharpe_ew", "Sharpe (EW Portfolio)", axes[1]),
]:
    for arch, marker in zip(arch_order, markers):
        sub = comp[comp["architecture"] == arch].sort_values("K")
        ax.plot(sub["K"], sub[metric], marker=marker, label=arch,
                color=colors_map[arch], linewidth=1.5, markersize=6)
    ax.set_xlabel("Number of Factors (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by Architecture and K")
    ax.legend(title="Architecture", fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

fig.suptitle("Model Comparison — R² and Sharpe by Architecture", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIGS}/model_comparison.pdf", bbox_inches="tight")
fig.savefig(f"{FIGS}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved model_comparison")

print(f"\nAll figures written to {FIGS}/")
