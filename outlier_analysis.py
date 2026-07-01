# =============================================================================
# outlier_analysis.py
# Exploratory analysis: How and when do outlier models arise in WTA3?
# Produces 6 figures saved to results/outlier_analysis/
# =============================================================================

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import median_abs_deviation

plt.style.use('default')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

RESULTS_DIR  = 'results'
OUT_DIR      = os.path.join(RESULTS_DIR, 'outlier_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES  = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
COLORS       = ['#4FC3F7', '#81C784', '#FFB74D', '#F06292', '#CE93D8', '#80DEEA']
MODEL_COLOR  = dict(zip(MODEL_NAMES, COLORS))

PRED_START_CYCLE = 61
TOTAL_CYCLES     = 213
N_PRED           = TOTAL_CYCLES - PRED_START_CYCLE + 1   # 153

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
with open(os.path.join(RESULTS_DIR, 'training_summary.json')) as f:
    training = json.load(f)

with open(os.path.join(RESULTS_DIR, 'all_predictions.json')) as f:
    predictions = json.load(f)

N_EXP = len(training)
print(f"  {N_EXP} experiments loaded")

# ── Build arrays ──────────────────────────────────────────────────────────────
# outlier_matrix[exp, model_idx] = True if that model was flagged in that experiment
outlier_matrix = np.zeros((N_EXP, len(MODEL_NAMES)), dtype=bool)
val_rmse_matrix = np.zeros((N_EXP, len(MODEL_NAMES)))
weight_matrix   = np.zeros((N_EXP, len(MODEL_NAMES)))

for i, row in enumerate(training):
    for j, m in enumerate(MODEL_NAMES):
        outlier_matrix[i, j]  = row['outliers'][m]
        val_rmse_matrix[i, j] = row['val_rmses'][m]
        weight_matrix[i, j]   = row['weights'][m]

# pred_matrix[exp, model_idx, cycle] = predicted RUL
pred_matrix = np.zeros((N_EXP, len(MODEL_NAMES), N_PRED))
ens_matrix  = np.zeros((N_EXP, N_PRED))
y_true      = None

for i, row in enumerate(predictions):
    for j, m in enumerate(MODEL_NAMES):
        pred_matrix[i, j, :] = row['test_predictions'][m]
    ens_matrix[i, :]  = row['ensemble_test_pred']
    if y_true is None:
        y_true = np.array(row['y_test'])

cycles = np.arange(PRED_START_CYCLE, TOTAL_CYCLES + 1)  # 61..213
# RUL decreases from cycle 61 onward — y_true[0] is RUL at cycle 61
print(f"  y_true RUL range: [{y_true.min():.0f}, {y_true.max():.0f}]")

# Error = prediction - true (signed)
error_matrix = pred_matrix - y_true[np.newaxis, np.newaxis, :]   # (exp, model, cycle)
abs_error_matrix = np.abs(error_matrix)

print(f"  Outlier counts per model: { {m: int(outlier_matrix[:, j].sum()) for j, m in enumerate(MODEL_NAMES)} }")
print()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Outlier rate per model (bar chart)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 1: Outlier rate per model...")
outlier_rate = outlier_matrix.mean(axis=0) * 100   # percent

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(MODEL_NAMES, outlier_rate,
              color=[MODEL_COLOR[m] for m in MODEL_NAMES],
              edgecolor='white', linewidth=0.5, alpha=0.9)
ax.axhline(outlier_rate.mean(), color='black', linestyle='--', linewidth=1,
           label=f'Mean = {outlier_rate.mean():.1f}%')
for bar, rate in zip(bars, outlier_rate):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, color='black')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Outlier Rate (%)', fontsize=12)
ax.set_title('Figure 1 — Outlier Rate per Model\n(% of 100 experiments flagged by MAD criterion)',
             fontsize=12, pad=12)
ax.set_ylim(0, max(outlier_rate) * 1.3)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_outlier_rate_by_model.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Val RMSE distributions: outlier vs non-outlier per model
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 2: Val RMSE distributions...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax = axes[j]
    out_mask  = outlier_matrix[:, j]
    rmse_out  = val_rmse_matrix[out_mask, j]
    rmse_in   = val_rmse_matrix[~out_mask, j]

    bins = np.linspace(val_rmse_matrix[:, j].min(),
                       val_rmse_matrix[:, j].max(), 25)

    ax.hist(rmse_in,  bins=bins, alpha=0.7, color=MODEL_COLOR[m],
            label=f'Inlier (n={len(rmse_in)})')
    ax.hist(rmse_out, bins=bins, alpha=0.7, color='#FF5252',
            label=f'Outlier (n={len(rmse_out)})')

    # MAD threshold lines
    median_r = np.median(val_rmse_matrix[:, j])
    mad      = median_abs_deviation(val_rmse_matrix[:, j])
    ax.axvline(median_r + 2.5 * mad, color='black', linestyle='--',
               linewidth=1, label='2.5×MAD threshold')
    ax.axvline(median_r - 2.5 * mad, color='black', linestyle='--', linewidth=1)

    ax.set_title(m, fontsize=11)
    ax.set_xlabel('Val RMSE', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)

fig.suptitle('Figure 2 — Validation RMSE: Inlier vs Outlier Models\n(per model, across 100 experiments)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 'fig2_val_rmse_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Mean absolute test error: outlier vs non-outlier across cycles
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 3: Prediction error across cycles...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax = axes[j]
    out_mask = outlier_matrix[:, j]

    if out_mask.sum() > 0:
        mae_out = abs_error_matrix[out_mask, j, :].mean(axis=0)
        ax.plot(cycles, mae_out, color='#FF5252', linewidth=1.5,
                label=f'Outlier runs (n={out_mask.sum()})', alpha=0.9)

    if (~out_mask).sum() > 0:
        mae_in  = abs_error_matrix[~out_mask, j, :].mean(axis=0)
        ax.plot(cycles, mae_in, color=MODEL_COLOR[m], linewidth=1.5,
                label=f'Inlier runs (n={(~out_mask).sum()})', alpha=0.9)

    ax.set_title(m, fontsize=11)
    ax.set_xlabel('Cycle', fontsize=9)
    ax.set_ylabel('Mean |Error| (cycles)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)
    ax.invert_xaxis()   # show degradation left-to-right as RUL decreases

fig.suptitle('Figure 3 — Mean Absolute Test Error: Outlier vs Inlier Runs\n(averaged across all flagged/non-flagged experiments, per model)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 'fig3_error_by_cycle_outlier_vs_inlier.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Heatmap: mean |error| by model × RUL bin
#            (inlier runs only — shows where each model naturally struggles)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 4: Heatmap model × RUL bin...")
rul_bins   = [0, 20, 50, 100, 150, 999]
rul_labels = ['0-20', '20-50', '50-100', '100-150', '>150']

heatmap_inlier  = np.zeros((len(MODEL_NAMES), len(rul_labels)))
heatmap_outlier = np.zeros((len(MODEL_NAMES), len(rul_labels)))
heatmap_count   = np.zeros((len(MODEL_NAMES), len(rul_labels)), dtype=int)

for b, (lo, hi) in enumerate(zip(rul_bins[:-1], rul_bins[1:])):
    mask_rul = (y_true >= lo) & (y_true < hi)
    for j in range(len(MODEL_NAMES)):
        in_mask  = ~outlier_matrix[:, j]
        out_mask =  outlier_matrix[:, j]
        if in_mask.sum() > 0 and mask_rul.sum() > 0:
            heatmap_inlier[j, b] = abs_error_matrix[in_mask, j, :][:, mask_rul].mean()
        if out_mask.sum() > 0 and mask_rul.sum() > 0:
            heatmap_outlier[j, b] = abs_error_matrix[out_mask, j, :][:, mask_rul].mean()
        heatmap_count[j, b] = mask_rul.sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, data, title in zip(
        axes,
        [heatmap_inlier, heatmap_outlier],
        ['Inlier Runs', 'Outlier Runs']):
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(rul_labels)))
    ax.set_xticklabels(rul_labels, fontsize=10)
    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_yticklabels(MODEL_NAMES, fontsize=10)
    ax.set_xlabel('True RUL Range (cycles)', fontsize=10)
    ax.set_title(f'Mean |Error| — {title}', fontsize=11)
    plt.colorbar(im, ax=ax, label='Mean |Error| (cycles)')
    for i in range(len(MODEL_NAMES)):
        for b in range(len(rul_labels)):
            ax.text(b, i, f'{data[i, b]:.1f}', ha='center', va='center',
                    fontsize=8, color='white' if data[i, b] > data.max()*0.5 else 'black')

fig.suptitle('Figure 4 — Mean Absolute Error by Model × RUL Range\n(left: inlier runs, right: outlier-flagged runs)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 'fig4_heatmap_model_rul.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Prediction bias: do outlier runs over- or under-estimate?
#            Signed error distribution for outlier vs inlier, all models combined
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5: Signed error / bias analysis...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax = axes[j]
    out_mask = outlier_matrix[:, j]

    if out_mask.sum() > 0:
        err_out = error_matrix[out_mask, j, :].flatten()
        ax.hist(err_out, bins=50, alpha=0.65, color='#FF5252',
                label=f'Outlier (n={out_mask.sum()})', density=True)
        ax.axvline(err_out.mean(), color='#c62828', linestyle='--',
                   linewidth=1.5, label=f'Outlier mean={err_out.mean():.1f}')

    if (~out_mask).sum() > 0:
        err_in = error_matrix[~out_mask, j, :].flatten()
        ax.hist(err_in, bins=50, alpha=0.65, color=MODEL_COLOR[m],
                label=f'Inlier (n={(~out_mask).sum()})', density=True)
        ax.axvline(err_in.mean(), color='#1565c0', linestyle='--',
                   linewidth=1.5, label=f'Inlier mean={err_in.mean():.1f}')

    ax.axvline(0, color='#555555', linewidth=0.8, alpha=0.5)
    ax.set_title(m, fontsize=11)
    ax.set_xlabel('Signed Error (pred - true)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)

fig.suptitle('Figure 5 — Signed Prediction Error: Bias Direction\n(positive = overestimate RUL, negative = underestimate)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 'fig5_signed_error_bias.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Trajectory smoothness: rolling variance of predictions
#            Outlier runs vs inlier runs — is there a detectable signal?
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 6: Trajectory smoothness (rolling variance)...")
WINDOW = 10

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax = axes[j]
    out_mask = outlier_matrix[:, j]

    def rolling_var(preds_2d, w=WINDOW):
        # preds_2d: (n_exps, n_cycles)
        rv = np.zeros_like(preds_2d)
        for c in range(preds_2d.shape[1]):
            lo = max(0, c - w + 1)
            rv[:, c] = preds_2d[:, lo:c+1].var(axis=1)
        return rv

    if out_mask.sum() > 0:
        rv_out = rolling_var(pred_matrix[out_mask, j, :])
        ax.plot(cycles, rv_out.mean(axis=0), color='#FF5252', linewidth=1.5,
                label=f'Outlier runs (n={out_mask.sum()})', alpha=0.9)
        ax.fill_between(cycles,
                        rv_out.mean(axis=0) - rv_out.std(axis=0),
                        rv_out.mean(axis=0) + rv_out.std(axis=0),
                        color='#FF5252', alpha=0.15)

    if (~out_mask).sum() > 0:
        rv_in = rolling_var(pred_matrix[~out_mask, j, :])
        ax.plot(cycles, rv_in.mean(axis=0), color=MODEL_COLOR[m], linewidth=1.5,
                label=f'Inlier runs (n={(~out_mask).sum()})', alpha=0.9)
        ax.fill_between(cycles,
                        rv_in.mean(axis=0) - rv_in.std(axis=0),
                        rv_in.mean(axis=0) + rv_in.std(axis=0),
                        color=MODEL_COLOR[m], alpha=0.1)

    ax.set_title(m, fontsize=11)
    ax.set_xlabel('Cycle', fontsize=9)
    ax.set_ylabel(f'Rolling Var (window={WINDOW})', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)

fig.suptitle('Figure 6 — Prediction Trajectory Smoothness: Outlier vs Inlier\n'
             f'(rolling variance over {WINDOW}-cycle window — higher = noisier trajectory)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 'fig6_trajectory_smoothness.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  OUTLIER ANALYSIS — SUMMARY STATISTICS")
print("=" * 60)
print()

total_flags = outlier_matrix.sum()
print(f"Total outlier flags (100 exp × 6 models = 600 possible): {total_flags} "
      f"({total_flags/600*100:.1f}%)")
print()

print(f"{'Model':<14} {'Outlier Rate':>13} {'Mean Val RMSE (outlier)':>24} "
      f"{'Mean Val RMSE (inlier)':>23} {'Mean |Error| (outlier)':>23} {'Mean |Error| (inlier)':>22}")
print("-" * 125)

for j, m in enumerate(MODEL_NAMES):
    out_mask  = outlier_matrix[:, j]
    rmse_out  = val_rmse_matrix[out_mask, j].mean()  if out_mask.sum()  > 0 else float('nan')
    rmse_in   = val_rmse_matrix[~out_mask, j].mean() if (~out_mask).sum() > 0 else float('nan')
    mae_out   = abs_error_matrix[out_mask,  j, :].mean() if out_mask.sum()  > 0 else float('nan')
    mae_in    = abs_error_matrix[~out_mask, j, :].mean() if (~out_mask).sum() > 0 else float('nan')
    print(f"  {m:<12} {out_mask.sum():>5}/100 ({out_mask.mean()*100:4.1f}%)   "
          f"{rmse_out:>20.2f}   {rmse_in:>20.2f}   "
          f"{mae_out:>20.2f}   {mae_in:>19.2f}")

print()

# Per-experiment: how many models are flagged?
n_flagged_per_exp = outlier_matrix.sum(axis=1)
print("Distribution of flagged models per experiment:")
for k in range(7):
    count = (n_flagged_per_exp == k).sum()
    print(f"  {k} models flagged: {count:3d} experiments ({count:.0f}%)")

print()

# RUL range where outlier error is highest
print("RUL ranges with highest outlier error (all models combined):")
for b, label in enumerate(rul_labels):
    mask_rul = (y_true >= rul_bins[b]) & (y_true < rul_bins[b+1])
    if mask_rul.sum() == 0:
        continue
    vals_out, vals_in = [], []
    for j in range(len(MODEL_NAMES)):
        om = outlier_matrix[:, j]
        if om.sum() > 0:
            vals_out.append(abs_error_matrix[om, j, :][:, mask_rul].mean())
        if (~om).sum() > 0:
            vals_in.append(abs_error_matrix[~om, j, :][:, mask_rul].mean())
    mean_out = np.mean(vals_out) if vals_out else float('nan')
    mean_in  = np.mean(vals_in)  if vals_in  else float('nan')
    print(f"  RUL {label:>8}: outlier |error|={mean_out:.2f}  inlier |error|={mean_in:.2f}  (n_cycles={mask_rul.sum()})")

print()
print(f"Figures saved to: {OUT_DIR}/")
print("  fig1_outlier_rate_by_model.png")
print("  fig2_val_rmse_distributions.png")
print("  fig3_error_by_cycle_outlier_vs_inlier.png")
print("  fig4_heatmap_model_rul.png")
print("  fig5_signed_error_bias.png")
print("  fig6_trajectory_smoothness.png")
