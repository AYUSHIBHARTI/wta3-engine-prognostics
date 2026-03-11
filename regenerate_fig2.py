# =============================================================================
# regenerate_fig2.py  —  PHM / IEEE TIE / MSSP journal standard style
# Clean, professional, easy to read
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import json, os

# ── Config ────────────────────────────────────────────────────────────────────
CALIBRATION_CYCLES = 60
TOTAL_CYCLES       = 213
N_EXPERIMENTS      = 100
RESULTS_DIR        = 'results'
FIGS_DIR           = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

PRED_START_CYCLE = CALIBRATION_CYCLES + 1
n_pred           = TOTAL_CYCLES - PRED_START_CYCLE + 1

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
with open(os.path.join(RESULTS_DIR, 'all_predictions.json')) as f:
    all_preds = json.load(f)
with open(os.path.join(RESULTS_DIR, 'final_metrics.json')) as f:
    final = json.load(f)

# ── Arrays ────────────────────────────────────────────────────────────────────
pred_cycles   = np.arange(PRED_START_CYCLE, TOTAL_CYCLES + 1)
true_rul      = np.arange(TOTAL_CYCLES - PRED_START_CYCLE, -1, -1)
full_cycles   = np.arange(1, TOTAL_CYCLES + 1)
full_true_rul = np.arange(TOTAL_CYCLES - 1, -1, -1)

ens_preds = np.array([e['ensemble_test_pred'] for e in all_preds])
meta_pred = ens_preds.mean(axis=0)
meta_std  = ens_preds.std(axis=0)

meta_rmse = final['meta_ensemble']['rmse']
meta_mae  = final['meta_ensemble']['mae']
meta_r2   = final['meta_ensemble']['r2']

# ── Uncertainty ───────────────────────────────────────────────────────────────
abs_errors = np.abs(ens_preds - true_rul[np.newaxis, :])
q_90       = np.percentile(abs_errors.mean(axis=1), 90)
residuals  = np.abs(meta_pred - true_rul)
valid_mask = meta_std > 0.1
k_90       = np.percentile(residuals[valid_mask] / meta_std[valid_mask], 90) \
             if valid_mask.sum() > 10 else 2.0

ci_low_var   = meta_pred - k_90 * meta_std
ci_high_var  = meta_pred + k_90 * meta_std
ci_low_conf  = meta_pred - q_90
ci_high_conf = meta_pred + q_90

cov_var  = np.mean((true_rul >= ci_low_var)  & (true_rul <= ci_high_var))  * 100
cov_conf = np.mean((true_rul >= ci_low_conf) & (true_rul <= ci_high_conf)) * 100

# ── Style — IEEE/MSSP standard ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.linewidth':    1.0,
    'axes.edgecolor':    'black',
    'axes.spines.top':   True,
    'axes.spines.right': True,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size':  4,
    'ytick.major.size':  4,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
})

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── Calibration zone — muted light grey ──────────────────────────────────────
ax.axvspan(1, CALIBRATION_CYCLES,
           color='#90CAF9', alpha=0.5, zorder=0)

# ── Variance CI — light blue fill + subtle blue outline (inner, drawn first) ─
ax.fill_between(pred_cycles, ci_low_var, ci_high_var,
                color='#90CAF9', alpha=0.55, zorder=2,
                edgecolor='#5BA3D9', linewidth=0.8)

# ── Conformal CI — rosy pink fill + subtle pink outline (outer, on top) ──────
ax.fill_between(pred_cycles, ci_low_conf, ci_high_conf,
                color='#F48FB1', alpha=0.45, zorder=3,
                edgecolor='#D95F8A', linewidth=0.8)

# ── True RUL — solid black ────────────────────────────────────────────────────
ax.plot(full_cycles, full_true_rul,
        color='black', linewidth=2.0, zorder=10,
        label='True RUL')

# ── Meta-ensemble — deep crimson ─────────────────────────────────────────────
ax.plot(pred_cycles, meta_pred,
        color='#C0392B', linewidth=2.0, zorder=9,
        label=f'WTA³ Ensemble (RMSE={meta_rmse:.2f}, MAE={meta_mae:.2f}, R²={meta_r2:.4f})')

# ── Prediction start — soft grey dashed vertical ─────────────────────────────
ax.axvline(CALIBRATION_CYCLES, color='#666666',
           linewidth=1.3, linestyle='--', zorder=8)

# ── Zone labels ───────────────────────────────────────────────────────────────
ax.text(30, full_true_rul.max() * 0.96,
        '', ha='center', va='top',
        fontsize=9, color='#777777', fontstyle='italic')
ax.text(CALIBRATION_CYCLES + 2, full_true_rul.max() * 0.96,
        '', ha='left', va='top',
        fontsize=9, color='#777777', fontstyle='italic')

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel('Cycle', fontsize=12)
ax.set_ylabel('RUL (cycles)', fontsize=12)
ax.set_title(
    'RUL Prediction with Uncertainty Interval on Representative Engine',
    fontsize=12, pad=10
)
ax.set_xlim(0, TOTAL_CYCLES + 3)
ax.set_ylim(-5, full_true_rul.max() + 14)
ax.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)

# ── Legend — minimal, inside plot ─────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color='black', linewidth=1.8,
           label='True RUL'),
    Line2D([0], [0], color='#C0392B', linewidth=1.8,
           label=f'WTA³ Ensemble  (RMSE={meta_rmse:.2f}, MAE={meta_mae:.2f}, R²={meta_r2:.4f})'),
    mpatches.Patch(facecolor='#F48FB1', edgecolor='#D95F8A', linewidth=0.8, alpha=0.65,
                   label=f'90% Conformal CI  (q$_{{90}}$=±{q_90:.1f} cyc, cov.={cov_conf:.0f}%)'),
    mpatches.Patch(facecolor='#90CAF9', edgecolor='#5BA3D9', linewidth=0.8, alpha=0.75,
                   label=f'90% Variance CI  (k$_{{90}}$={k_90:.2f}σ, cov.={cov_var:.0f}%)'),
]

ax.legend(handles=legend_elements,
          fontsize=9, loc='upper right',
          bbox_to_anchor=(1, 0.93),
          frameon=True, framealpha=0.9,
          edgecolor='#aaaaaa', borderpad=0.7,
          handlelength=1.8, labelspacing=0.4)

plt.tight_layout(pad=1.2)
p = os.path.join(FIGS_DIR, 'fig2_main_rul_prediction.png')
plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"  Saved: {p}")
print("Done!")