import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

RESULTS = "./results"

# ── 1. LOAD CSV ───────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(RESULTS, "meta_ensemble_predictions.csv"))

cycles_pred   = df["cycle"].values
true_pred_w   = df["true_RUL"].values
meta_pred     = df["meta_ensemble_pred"].values
epistemic_std = df["meta_ensemble_std"].values
var_lower     = df["CI_var_low"].values
var_upper     = df["CI_var_high"].values

# ── 2. CONFORMAL CI — load q90 from final_metrics.json ───────────────────────
with open(os.path.join(RESULTS, "final_metrics.json"), "r") as f:
    final = json.load(f)

print("final_metrics keys:", list(final.keys()))

# Try common key names — will print what it finds
conformal_q90 = None
for key in ["conformal_q90", "q90", "conf_q90", "conformal_halfwidth"]:
    if key in final:
        conformal_q90 = final[key]
        print(f"Found conformal_q90 under key '{key}': {conformal_q90}")
        break

if conformal_q90 is None:
    conformal_q90 = 5.8   # fallback from your Figure 6 title
    print(f"Key not found — using fallback conformal_q90={conformal_q90}")

k90 = None
for key in ["k90", "k_90", "var_k90", "calibration_factor"]:
    if key in final:
        k90 = final[key]
        print(f"Found k90 under key '{key}': {k90}")
        break

if k90 is None:
    k90 = 3.04
    print(f"Key not found — using fallback k90={k90}")

conf_lower = np.clip(meta_pred - conformal_q90, 0, None)
conf_upper = meta_pred + conformal_q90

# ── 3. FULL RANGE ENVELOPE from all_predictions.json ─────────────────────────
has_range = False
try:
    with open(os.path.join(RESULTS, "all_predictions.json"), "r") as f:
        all_data = json.load(f)
    print("\nall_predictions.json top-level keys:", list(all_data.keys())[:15])

    # Try to find experiment-level predictions
    for key in ["ensemble_predictions", "predictions", "all_preds", "experiments"]:
        if key in all_data:
            exp_preds = np.array(all_data[key])
            print(f"Found '{key}' with shape: {exp_preds.shape}")
            if exp_preds.ndim == 2:
                range_lower = np.min(exp_preds, axis=0)
                range_upper = np.max(exp_preds, axis=0)
                has_range = True
            break
except Exception as e:
    print(f"Range envelope skipped: {e}")

# ── 4. TRUE RUL full lifecycle (cycles 1-213) ─────────────────────────────────
total_cycles  = int(cycles_pred[-1])          # 213
cycles_full   = np.arange(1, total_cycles + 1)
true_rul_full = total_cycles - cycles_full    # 212 down to 0

# ── 5. COLORS ─────────────────────────────────────────────────────────────────
C_TRUE  = "#1a1a1a"
C_PRED  = "#c0392b"
C_RANGE = "#bdc3c7"
C_VAR   = "#2980b9"
C_CONF  = "#e67e22"
C_CALBG = "#ebf5fb"
cal_end = 60

# ── 6. PLOT ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))

ax.axvspan(1, cal_end, color=C_CALBG, alpha=0.6, zorder=0,
           label="Calibration window (cycles 1–60)")

if has_range:
    ax.fill_between(cycles_pred, range_lower, range_upper,
                    color=C_RANGE, alpha=0.25, zorder=1,
                    label="Full prediction range (100 experiments)")

# Conformal CI — draw as bordered outline so it stays visible inside variance CI
ax.fill_between(cycles_pred, conf_lower, conf_upper,
                color=C_CONF, alpha=0.15, zorder=2)
ax.plot(cycles_pred, conf_upper, color=C_CONF, lw=1.2, ls="-", zorder=2, alpha=0.85)
ax.plot(cycles_pred, conf_lower, color=C_CONF, lw=1.2, ls="-", zorder=2, alpha=0.85,
        label=f"90% Conformal CI  (\u00b1{conformal_q90:.1f} cycles, cov.=88%\u2020)")

ax.fill_between(cycles_pred, var_lower, var_upper,
                color=C_VAR, alpha=0.28, zorder=3,
                label=f"90% Variance CI  (k\u2089\u2080={k90:.2f}\u03c3, cov.=100%)")

ax.plot(cycles_full, true_rul_full,
        color=C_TRUE, lw=2.0, zorder=5, label="True RUL")

ax.plot(cycles_pred, meta_pred,
        color=C_PRED, lw=1.8, zorder=6,
        label="WTA\u00b3 Meta-Ensemble  (RMSE=3.73, MAE=3.19, R\u00b2=0.9929)")

ax.axvline(cal_end, color="#f39c12", lw=1.5, ls="--", zorder=4,
           label="Prediction start (cycle 60)")

for cyc, lbl in [(111, "Early | Mid"), (162, "Mid | Late")]:
    ax.axvline(cyc, color="#7f8c8d", lw=1.0, ls=":", zorder=4)
    ax.text(cyc - 2.5, true_rul_full.max() * 0.96,
            lbl, fontsize=9, color="#7f8c8d", va="top", ha="right")

# ── 7. FORMATTING ─────────────────────────────────────────────────────────────
ax.set_xlabel("Cycle", labelpad=8)
ax.set_ylabel("RUL (cycles)", labelpad=8)
ax.set_xlim(1, total_cycles + 3)
ax.set_ylim(-5, true_rul_full.max() + 15)
ax.legend(loc="upper right", framealpha=0.93,
          edgecolor="#cccccc", handlelength=1.8)

fig.text(
    0.13, 0.01,
    ("\u2020 88% empirical coverage on Engine #52 is consistent with the marginal nature "
     "of the conformal guarantee (holds in expectation over repeated test draws, "
     "not on every individual engine trajectory)."),
    fontsize=8, color="#555555", style="italic"
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

# ── 8. SAVE ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
out_pdf = os.path.join(RESULTS, "figures", "figure2_final.pdf")
out_png = os.path.join(RESULTS, "figures", "figure2_final.png")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_png}")
plt.show()