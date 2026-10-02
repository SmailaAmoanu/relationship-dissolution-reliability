from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from common import PROJECT, RESULTS

OUT = PROJECT / "figures"
OUT.mkdir(exist_ok=True)

bench = pd.read_csv(PROJECT / "results" / "reference" / "model_benchmark_metrics.csv")
stab = pd.read_csv(RESULTS / "06_reliability_analysis" / "03_calibration_conformal_diagnostics.csv")
ranks = pd.read_csv(RESULTS / "04_robustness" / "full_refit_importance_ranks.csv")

# Figure 1: held-out Random Forest discrimination
rf = bench[bench["model"] == "Random Forest"].copy()
x = np.arange(2)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
vals = rf["roc_auc"].to_numpy()
lo = vals - rf["roc_ci_low"].to_numpy()
hi = rf["roc_ci_high"].to_numpy() - vals
ax.errorbar(x, vals, yerr=np.vstack([lo, hi]), fmt="o", capsize=4, linewidth=1.2)
ax.set_xticks(x, ["3-year", "5-year"])
ax.set_ylabel("ROC-AUC")
ax.set_xlabel("Forecast horizon")
ax.set_ylim(0.78, 0.97)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "Figure_1_RF_discrimination.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 2: explanation stability across repeated refits
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for pos, (horizon, marker) in enumerate([("3-year", "o"), ("5-year", "s")]):
    z = ranks[ranks.horizon == horizon].pivot(index="variable", columns="seed", values="rank")
    seeds = list(z.columns)
    values = [spearmanr(z[a], z[b]).statistic for i, a in enumerate(seeds) for b in seeds[i+1:]]
    ax.scatter(np.full(len(values), pos), values, marker=marker, s=22, alpha=.55)
    ax.plot(pos, np.mean(values), marker="_", markersize=22, mew=2)
ax.set_xticks([0, 1], ["3-year", "5-year"])
ax.set_ylabel("Pairwise Spearman rank correlation")
ax.set_xlabel("Forecast horizon")
ax.set_ylim(0.88, 1.005)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "Figure_2_explanation_stability.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 3: outcome-specific conformal coverage
fig, ax = plt.subplots(figsize=(7.0, 4.5))
positions = np.arange(4)
labels, cont, diss = [], [], []
for horizon in ["3-year", "5-year"]:
    for method in ["Pooled", "Class-conditional"]:
        row = stab[(stab.horizon == horizon) & (stab.method == method)].iloc[0]
        labels.append(f"{horizon}\n{method}")
        cont.append(row.continuation_coverage)
        diss.append(row.dissolution_coverage)
w = .34
ax.bar(positions-w/2, cont, width=w, label="Continuation")
ax.bar(positions+w/2, diss, width=w, label="Dissolution")
ax.axhline(.90, linestyle="--", linewidth=1)
ax.set_ylabel("Empirical coverage")
ax.set_xticks(positions, labels)
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "Figure_3_conformal_class_coverage.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 4: minority-class coverage versus decisiveness
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for horizon, marker in [("3-year", "o"), ("5-year", "s")]:
    z = stab[stab.horizon == horizon]
    ax.scatter(z["singleton_rate"], z["dissolution_coverage"], marker=marker, s=55, label=horizon)
    for _, row in z.iterrows():
        ax.annotate(row["method"], (row["singleton_rate"], row["dissolution_coverage"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
ax.axhline(.90, linestyle="--", linewidth=1)
ax.set_xlabel("Singleton prediction-set rate")
ax.set_ylabel("Dissolution-class coverage")
ax.set_xlim(.65, 1.02)
ax.set_ylim(0, 1.02)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "Figure_4_uncertainty_tradeoff.pdf", bbox_inches="tight")
plt.close(fig)
