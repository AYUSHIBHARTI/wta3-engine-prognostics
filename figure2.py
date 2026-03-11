import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})

# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("./results/meta_ensemble_predictions.csv")
cycles   = df["cycle"].values            # 61 to 213
true_rul = df["true_RUL"].values
pred     = df["meta_ensemble_pred"].values
var_lo   = df["CI_var_low"].values
var_hi   = df["CI_var_high"].values

# Conformal CI: fixed half-width from your calibration
q90 = 5.8
conf_lo = np.clip(pred - q90, 0, None)
conf_hi = pred + q90

# Full true RUL lifecycle (cycles 1 to 213)
total    = int(cycles[-1])
cyc_full = np.arange(1, total + 1)
rul_full = total - cyc_full

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))

# Calibration zone
ax.axvspan(1, 60, color="#ebf5fb", alpha=0.7, zorder=0,
           label="Calibration window (cycles 1–60)")

# Variance CI — blue filled band
ax.fill_between(cycles, var_lo, var_hi,
                color="#2980b9", alpha=0.25, zorder=2,
                label="90% Variance CI  (k\u2089\u2080=3.04\u03c3, cov.=100%)")

# Conformal CI — orange boundary lines (always visible)
ax.fill_between(cycles, conf_lo, conf_hi,
                color="#e67e22", alpha=0.10, zorder=3)
ax.plot(cycles, conf_hi, color="#e67e22", lw=1.4, zorder=3)
ax.plot(cycles, conf_lo, color="#e67e22", lw=1.4, zorder=3,
        label="90% Conformal CI  (\u00b15.8 cycles, cov.=88%\u2020)")

# True RUL
ax.plot(cyc_full, rul_full, color="#1a1a1a", lw=2.0, zorder=5,
        label="True RUL")

# Prediction
ax.plot(cycles, pred, color="#c0392b", lw=1.8, zorder=6,
        label="WTA\u00b3 Meta-Ensemble  (RMSE=3.73, MAE=3.19, R\u00b2=0.9929)")

# Prediction start
ax.axvline(60, color="#f39c12", lw=1.5, ls="--", zorder=4,
           label="Prediction start (cycle 60)")

# Phase boundaries — labels LEFT of line
for cyc, lbl in [(111, "Early | Mid"), (162, "Mid | Late")]:
    ax.axvline(cyc, color="#7f8c8d", lw=1.0, ls=":", zorder=4)
    ax.text(cyc - 2, rul_full.max() * 0.97, lbl,
            fontsize=9, color="#7f8c8d", ha="right", va="top")

# ── FORMAT ────────────────────────────────────────────────────────────────────
ax.set_xlabel("Cycle", labelpad=8)
ax.set_ylabel("RUL (cycles)", labelpad=8)
ax.set_xlim(1, total + 3)
ax.set_ylim(-5, rul_full.max() + 15)
ax.legend(loc="upper right", framealpha=0.93, edgecolor="#cccccc")

fig.text(0.13, 0.01,
    "\u2020 88% empirical coverage on Engine #52 is consistent with the marginal "
    "nature of the conformal guarantee (holds in expectation over repeated test "
    "draws, not on every individual engine trajectory).",
    fontsize=8, color="#555555", style="italic")

fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("./results/figures/figure2_final.png", dpi=300, bbox_inches="tight")
fig.savefig("./results/figures/figure2_final.pdf", dpi=300, bbox_inches="tight")
print("Done. Check ./results/figures/figure2_final.png")
plt.show()