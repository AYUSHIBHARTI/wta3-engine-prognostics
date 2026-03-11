# =============================================================================
# step6_visualization_v5.py
# Changes from v4:
#   - Conformal CI (q90) removed from all figures and metrics
#   - Variance CI (k90=3.04σ) is now the sole uncertainty interval
#   - Figure 2 and Figure 6 updated
#   - final_metrics.json updated (no q90 field)
# All other analyses identical to v4
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os
from scipy.stats import median_abs_deviation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION_CYCLES = 60
SEQUENCE_LENGTH    = 50
TOTAL_CYCLES       = 213
N_EXPERIMENTS      = 100
MODEL_NAMES        = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
RESULTS_DIR        = 'results'
FIGS_DIR           = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

PRED_START_CYCLE = CALIBRATION_CYCLES + 1   # 61
n_pred           = TOTAL_CYCLES - PRED_START_CYCLE + 1  # 153
N_PRED           = n_pred

colors6 = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL SAVED DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading saved results...")

with open(os.path.join(RESULTS_DIR, 'all_predictions.json'))      as f: all_preds     = json.load(f)
with open(os.path.join(RESULTS_DIR, 'training_summary.json'))     as f: train_summary = json.load(f)
with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json'))   as f: eval_summary  = json.load(f)
with open(os.path.join(RESULTS_DIR, 'aggregate_metrics.json'))    as f: agg_metrics   = json.load(f)
with open(os.path.join(RESULTS_DIR, 'step1_checkpoint.json'))     as f: step1         = json.load(f)
with open(os.path.join(RESULTS_DIR, 'step2_checkpoint.json'))     as f: step2         = json.load(f)
with open(os.path.join(RESULTS_DIR, 'val_calibration_data.json')) as f: val_cal       = json.load(f)

# ── Variance CI only — conformal CI dropped ───────────────────────────────────
k_90         = val_cal['k90']
n_cal_cycles = val_cal['n_val_cycles']

print(f"  Variance CI: k90={k_90:.4f}  calibrated on {n_cal_cycles} val cycles")
print(f"  Note: Conformal CI dropped — q90=118 cycles was uninformative")
print(f"        due to fleet heterogeneity in validation residuals")

# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCT ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
pred_cycles   = np.arange(PRED_START_CYCLE, TOTAL_CYCLES + 1)
true_rul      = np.arange(TOTAL_CYCLES - PRED_START_CYCLE, -1, -1)
full_cycles   = np.arange(1, TOTAL_CYCLES + 1)
full_true_rul = np.arange(TOTAL_CYCLES - 1, -1, -1)

print(f"\n  Prediction period: cycles {PRED_START_CYCLE}–{TOTAL_CYCLES} ({n_pred} points)")

ens_preds = np.array([e['ensemble_test_pred'] for e in all_preds])
assert ens_preds.shape == (N_EXPERIMENTS, n_pred), \
    f"Shape mismatch: {ens_preds.shape}"

meta_pred = ens_preds.mean(axis=0)
meta_std  = ens_preds.std(axis=0)

print(f"  Loaded {len(all_preds)} experiment predictions")
print(f"  Meta-ensemble shape: {meta_pred.shape}")

# ── Variance CI bands ─────────────────────────────────────────────────────────
ci_low_var  = meta_pred - k_90 * meta_std
ci_high_var = meta_pred + k_90 * meta_std
cov_var     = float(np.mean((true_rul >= ci_low_var) & (true_rul <= ci_high_var)) * 100)

print(f"  Variance CI coverage: {cov_var:.1f}%  (nominal 90%)")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE MASKS
# ─────────────────────────────────────────────────────────────────────────────
early_m = (pred_cycles >= 61)  & (pred_cycles <= 111)
mid_m   = (pred_cycles >= 112) & (pred_cycles <= 162)
late_m  = (pred_cycles >= 163) & (pred_cycles <= 213)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def metrics(pred, true):
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae  = float(mean_absolute_error(true, pred))
    r2   = float(r2_score(true, pred))
    nasa = float(np.sum(np.where(
        pred - true < 0,
        np.exp(-(pred - true) / 13.0) - 1,
        np.exp( (pred - true) / 10.0) - 1)))
    w10  = float(np.mean(np.abs(true - pred) <= 10) * 100)
    return rmse, mae, r2, nasa, w10

def nasa_score_func(pred, true):
    e = np.array(pred) - np.array(true)
    return float(np.sum(np.where(e < 0,
                                 np.exp(-e / 13.0) - 1,
                                 np.exp( e / 10.0) - 1)))

meta_rmse, meta_mae, meta_r2, meta_nasa, meta_w10 = metrics(meta_pred, true_rul)

print(f"\n  Meta-ensemble: RMSE={meta_rmse:.2f}  MAE={meta_mae:.2f}  "
      f"R²={meta_r2:.4f}  NASA={meta_nasa:.2f}  W10={meta_w10:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE-WISE METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing phase-wise metrics...")

meta_mae_e = float(mean_absolute_error(true_rul[early_m], meta_pred[early_m]))
meta_mae_m = float(mean_absolute_error(true_rul[mid_m],   meta_pred[mid_m]))
meta_mae_l = float(mean_absolute_error(true_rul[late_m],  meta_pred[late_m]))

meta_nasa_e = nasa_score_func(meta_pred[early_m], true_rul[early_m])
meta_nasa_m = nasa_score_func(meta_pred[mid_m],   true_rul[mid_m])
meta_nasa_l = nasa_score_func(meta_pred[late_m],  true_rul[late_m])

print(f"  Early MAE={meta_mae_e:.4f}  Mid MAE={meta_mae_m:.4f}  Late MAE={meta_mae_l:.4f}")
print(f"  Early NASA={meta_nasa_e:.4f}  Mid NASA={meta_nasa_m:.4f}  Late NASA={meta_nasa_l:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISM BIAS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\nOptimism Bias Analysis...")

val_rmses_exp = np.array([e['ensemble_val']['rmse'] for e in eval_summary])
sorted_idx    = np.argsort(val_rmses_exp)
top50_idx     = sorted_idx[:50]
bot50_idx     = sorted_idx[50:]

top50_meta = ens_preds[top50_idx].mean(axis=0)
bot50_meta = ens_preds[bot50_idx].mean(axis=0)

top50_rmse = float(np.sqrt(mean_squared_error(true_rul, top50_meta)))
bot50_rmse = float(np.sqrt(mean_squared_error(true_rul, bot50_meta)))
top50_mae  = float(mean_absolute_error(true_rul, top50_meta))
bot50_mae  = float(mean_absolute_error(true_rul, bot50_meta))
top50_r2   = float(r2_score(true_rul, top50_meta))
bot50_r2   = float(r2_score(true_rul, bot50_meta))
bias_ratio = (bot50_rmse - top50_rmse) / meta_rmse * 100

if   abs(bot50_rmse - top50_rmse) < 0.5: bias_verdict = "NEGLIGIBLE"
elif abs(bot50_rmse - top50_rmse) < 1.5: bias_verdict = "MODERATE"
else:                                     bias_verdict = "SIGNIFICANT"

print(f"  Top-50 val → test RMSE: {top50_rmse:.2f}")
print(f"  Bot-50 val → test RMSE: {bot50_rmse:.2f}")
print(f"  All-100    → test RMSE: {meta_rmse:.2f}")
print(f"  Gap: {bot50_rmse - top50_rmse:.2f} cycles ({bias_ratio:.1f}% of meta RMSE)")
print(f"  Verdict: {bias_verdict}")

optimism_bias_result = {
    'top50_rmse'      : top50_rmse,
    'bot50_rmse'      : bot50_rmse,
    'all100_rmse'     : meta_rmse,
    'gap_cycles'      : float(bot50_rmse - top50_rmse),
    'gap_pct_of_meta' : float(bias_ratio),
    'verdict'         : bias_verdict,
}

# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL MEAN METRICS
# ─────────────────────────────────────────────────────────────────────────────
per_model_test = {m: {'rmse': [], 'mae': [], 'nasa': [], 'r2': []} for m in MODEL_NAMES}
for e in eval_summary:
    for m in MODEL_NAMES:
        per_model_test[m]['rmse'].append(e['per_model_test'][m]['rmse'])
        per_model_test[m]['mae'].append( e['per_model_test'][m]['mae'])
        per_model_test[m]['nasa'].append(e['per_model_test'][m]['nasa_score'])
        per_model_test[m]['r2'].append(  e['per_model_test'][m]['r2'])

per_model_mean = {m: {k: float(np.mean(v)) for k, v in per_model_test[m].items()}
                  for m in MODEL_NAMES}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL PREDICTIONS FOR SENSITIVITY + ORACLE
# ─────────────────────────────────────────────────────────────────────────────
all_val_rmses = [{m: exp['val_rmses'][m] for m in MODEL_NAMES}
                 for exp in train_summary]

all_model_preds = {m: [] for m in MODEL_NAMES}
for exp in all_preds:
    for m in MODEL_NAMES:
        all_model_preds[m].append(np.array(exp['test_predictions'][m]))
for m in MODEL_NAMES:
    all_model_preds[m] = np.array(all_model_preds[m])  # (100, 153)

# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def compute_wta3_weights_param(val_rmses_dict, model_names,
                               alpha=0.05, beta=-3, mad_threshold=2.5):
    vals     = np.array([val_rmses_dict[m] for m in model_names], dtype=float)
    median_r = np.median(vals)
    mad      = float(median_abs_deviation(vals))

    if mad < 1e-10:
        outlier = np.zeros(len(vals), dtype=bool)
    else:
        outlier = np.abs(vals - median_r) > mad_threshold * mad

    inlier_vals = vals[~outlier]
    mean_inlier = float(np.mean(inlier_vals)) if len(inlier_vals) > 0 \
                  else float(np.mean(vals))

    raw_weights = np.zeros(len(vals))
    for i, r in enumerate(vals):
        if not outlier[i]:
            raw_weights[i] = (r + alpha * mean_inlier) ** beta

    total = raw_weights.sum()
    if total > 0:
        raw_weights /= total
    else:
        n_inliers = (~outlier).sum()
        for i in range(len(vals)):
            if not outlier[i]:
                raw_weights[i] = 1.0 / n_inliers if n_inliers > 0 else 0.0

    weights  = {m: float(raw_weights[i]) for i, m in enumerate(model_names)}
    outliers = {m: bool(outlier[i])      for i, m in enumerate(model_names)}
    return weights, outliers, int(outlier.sum())

# =============================================================================
# SENSITIVITY A — β Exponent Sweep
# =============================================================================
print("\n" + "="*70)
print("  SENSITIVITY A: β Exponent Sweep")
print("="*70)

beta_values  = [-1, -2, -3, -4, -5]
beta_results = []
MAD_FIXED    = 2.5

print(f"\n  {'β':>4}  {'RMSE':>7}  {'MAE':>7}  {'R²':>8}  "
      f"{'CNN_w':>7}  {'Transformer_w':>13}  {'SVR_w':>7}")
print("  " + "-"*65)

for beta in beta_values:
    exp_ens_preds = []
    mean_weights  = {m: [] for m in MODEL_NAMES}

    for exp_idx in range(N_EXPERIMENTS):
        weights, outliers, n_excl = compute_wta3_weights_param(
            all_val_rmses[exp_idx], MODEL_NAMES,
            alpha=0.05, beta=beta, mad_threshold=MAD_FIXED)

        ens = np.zeros(N_PRED)
        for m in MODEL_NAMES:
            ens += weights[m] * all_model_preds[m][exp_idx]
        exp_ens_preds.append(ens)

        for m in MODEL_NAMES:
            mean_weights[m].append(weights[m])

    exp_ens_preds = np.array(exp_ens_preds)
    meta_beta     = exp_ens_preds.mean(axis=0)

    rmse_b = float(np.sqrt(mean_squared_error(true_rul, meta_beta)))
    mae_b  = float(mean_absolute_error(true_rul, meta_beta))
    r2_b   = float(r2_score(true_rul, meta_beta))
    mean_w = {m: float(np.mean(mean_weights[m])) for m in MODEL_NAMES}
    marker = "  ← SELECTED" if beta == -3 else ""

    print(f"  {beta:>4}  {rmse_b:>7.4f}  {mae_b:>7.4f}  {r2_b:>8.4f}  "
          f"{mean_w['CNN']:>7.3f}  {mean_w['Transformer']:>13.3f}  "
          f"{mean_w['SVR']:>7.3f}{marker}")

    beta_results.append({
        'beta'        : beta,
        'rmse'        : rmse_b,
        'mae'         : mae_b,
        'r2'          : r2_b,
        'mean_weights': mean_w,
        'selected'    : (beta == -3),
    })

with open(os.path.join(RESULTS_DIR, 'sensitivity_beta.json'), 'w') as f:
    json.dump(beta_results, f, indent=2)
print(f"\n  Saved: results/sensitivity_beta.json")

# =============================================================================
# SENSITIVITY B — MAD Threshold Sweep
# =============================================================================
print("\n" + "="*70)
print("  SENSITIVITY B: MAD Threshold Sweep")
print("="*70)

mad_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
mad_results    = []
BETA_FIXED     = -3

print(f"\n  {'Threshold':>9}  {'RMSE':>7}  {'MAE':>7}  {'R²':>8}  "
      f"{'Transformer_excl':>16}  {'XGB_excl':>8}  {'Zero-model':>10}")
print("  " + "-"*75)

for threshold in mad_thresholds:
    exp_ens_preds      = []
    model_excl_cnt     = {m: 0 for m in MODEL_NAMES}
    zero_model_configs = 0

    for exp_idx in range(N_EXPERIMENTS):
        weights, outliers, n_excl = compute_wta3_weights_param(
            all_val_rmses[exp_idx], MODEL_NAMES,
            alpha=0.05, beta=BETA_FIXED, mad_threshold=threshold)

        for m in MODEL_NAMES:
            if outliers[m]:
                model_excl_cnt[m] += 1

        if all(weights[m] == 0 for m in MODEL_NAMES):
            zero_model_configs += 1
            for m in MODEL_NAMES:
                weights[m] = 1.0 / len(MODEL_NAMES)

        ens = np.zeros(N_PRED)
        for m in MODEL_NAMES:
            ens += weights[m] * all_model_preds[m][exp_idx]
        exp_ens_preds.append(ens)

    exp_ens_preds = np.array(exp_ens_preds)
    meta_mad      = exp_ens_preds.mean(axis=0)

    rmse_m     = float(np.sqrt(mean_squared_error(true_rul, meta_mad)))
    mae_m      = float(mean_absolute_error(true_rul, meta_mad))
    r2_m       = float(r2_score(true_rul, meta_mad))
    trans_excl = model_excl_cnt['Transformer']
    xgb_excl   = model_excl_cnt['XGB']
    marker     = "  ← SELECTED" if threshold == 2.5 else ""

    print(f"  {threshold:>9.1f}  {rmse_m:>7.4f}  {mae_m:>7.4f}  {r2_m:>8.4f}  "
          f"{trans_excl:>16}/100  {xgb_excl:>8}/100  "
          f"{zero_model_configs:>10}{marker}")

    mad_results.append({
        'threshold'         : threshold,
        'rmse'              : rmse_m,
        'mae'               : mae_m,
        'r2'                : r2_m,
        'transformer_excl'  : trans_excl,
        'xgb_excl'          : xgb_excl,
        'zero_model_configs': zero_model_configs,
        'all_model_excl'    : {m: model_excl_cnt[m] for m in MODEL_NAMES},
        'selected'          : (threshold == 2.5),
    })

with open(os.path.join(RESULTS_DIR, 'sensitivity_mad.json'), 'w') as f:
    json.dump(mad_results, f, indent=2)
print(f"\n  Saved: results/sensitivity_mad.json")

# =============================================================================
# ORACLE PHASE-ADAPTIVE GAP
# =============================================================================
print("\n" + "="*70)
print("  ORACLE PHASE-ADAPTIVE GAP")
print("="*70)

phase_masks      = {'early': early_m, 'mid': mid_m, 'late': late_m}
phase_labels     = {'early': 'Early (61–111)', 'mid': 'Mid (112–162)', 'late': 'Late (163–213)'}
meta_phase_maes  = {'early': meta_mae_e, 'mid': meta_mae_m, 'late': meta_mae_l}
oracle_results   = {}

print(f"\n  {'Phase':<18}  {'Best Model':<12}  {'Oracle MAE':>10}  "
      f"{'Meta MAE':>9}  {'Gap':>7}  {'Gap %':>7}")
print("  " + "-"*65)

for phase, mask in phase_masks.items():
    true_phase       = true_rul[mask]
    model_phase_maes = {}

    for m in MODEL_NAMES:
        phase_preds  = all_model_preds[m][:, mask]
        per_exp_mae  = np.array([
            mean_absolute_error(true_phase, phase_preds[i])
            for i in range(N_EXPERIMENTS)
        ])
        model_phase_maes[m] = float(np.mean(per_exp_mae))

    best_model     = min(model_phase_maes, key=model_phase_maes.get)
    oracle_mae     = model_phase_maes[best_model]
    meta_mae_phase = meta_phase_maes[phase]
    gap            = meta_mae_phase - oracle_mae
    gap_pct        = (gap / meta_mae_phase) * 100 if meta_mae_phase > 0 else 0.0

    oracle_results[phase] = {
        'best_model'    : best_model,
        'oracle_mae'    : oracle_mae,
        'meta_mae'      : meta_mae_phase,
        'gap_cycles'    : float(gap),
        'gap_pct'       : float(gap_pct),
        'all_model_maes': model_phase_maes,
    }
    print(f"  {phase_labels[phase]:<18}  {best_model:<12}  "
          f"{oracle_mae:>10.4f}  {meta_mae_phase:>9.4f}  "
          f"{gap:>7.4f}  {gap_pct:>6.1f}%")

oracle_combined_mae = float(np.mean([oracle_results[p]['oracle_mae'] for p in phase_masks]))
meta_combined_mae   = float(np.mean([meta_mae_e, meta_mae_m, meta_mae_l]))
total_gap           = meta_combined_mae - oracle_combined_mae
total_gap_pct       = (total_gap / meta_combined_mae) * 100 if meta_combined_mae > 0 else 0.0

print(f"\n  Oracle combined MAE : {oracle_combined_mae:.4f}")
print(f"  Meta combined MAE   : {meta_combined_mae:.4f}")
print(f"  Total gap           : {total_gap:.4f} cycles ({total_gap_pct:.1f}%)")

oracle_summary = {
    'per_phase'           : oracle_results,
    'oracle_combined_mae' : oracle_combined_mae,
    'meta_combined_mae'   : meta_combined_mae,
    'total_gap_cycles'    : float(total_gap),
    'total_gap_pct'       : float(total_gap_pct),
    'note': (
        'Oracle = best single model per phase averaged across 100 experiments. '
        'Post-hoc only — phase labels unavailable at inference time.'
    ),
}
with open(os.path.join(RESULTS_DIR, 'oracle_gap.json'), 'w') as f:
    json.dump(oracle_summary, f, indent=2)
print(f"  Saved: results/oracle_gap.json")

# =============================================================================
# FIGURE 1 — Experiment RMSE Distribution
# =============================================================================
print("\nGenerating Figure 1...")

exp_rmses = np.array([e['ensemble_test']['rmse'] for e in eval_summary])
exp_maes  = np.array([e['ensemble_test']['mae']  for e in eval_summary])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(exp_rmses, bins=20, color='#3498db', edgecolor='black', alpha=0.8)
axes[0].axvline(meta_rmse, color='#e74c3c', lw=2.5, ls='--',
                label=f'Meta-Ensemble RMSE={meta_rmse:.2f}')
axes[0].axvline(np.median(exp_rmses), color='#f39c12', lw=2, ls=':',
                label=f'Median={np.median(exp_rmses):.2f}')
axes[0].set_xlabel('Ensemble RMSE (cycles)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution of WTA³ Ensemble RMSE\nacross 100 Experiments',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

bp = axes[1].boxplot([exp_rmses, exp_maes], labels=['RMSE', 'MAE'],
                     patch_artist=True, notch=True)
for patch, c in zip(bp['boxes'], ['#3498db', '#e74c3c']):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
axes[1].set_ylabel('Error (cycles)', fontsize=12, fontweight='bold')
axes[1].set_title('RMSE & MAE Distribution\nacross 100 Experiments',
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Figure 1: Statistical Distribution of WTA³ Ensemble Performance',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig1_experiment_distribution.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig1")



# =============================================================================
# FIGURE 2 — Main RUL Prediction (variance CI only)
# =============================================================================
print("Generating Figure 2...")

fig, ax = plt.subplots(figsize=(18, 8))

ax.fill_between(pred_cycles, ens_preds.min(axis=0), ens_preds.max(axis=0),
                color='#d9d9d9', alpha=0.5, zorder=1,
                label='Full Range (100 ensembles)')
ax.fill_between(pred_cycles, ci_low_var, ci_high_var,
                color='#3498db', alpha=0.30, zorder=3,
                label=f'90% Variance CI  k90={k_90:.2f}σ  cov={cov_var:.0f}%')
ax.plot(full_cycles, full_true_rul, 'k-', lw=3.5, label='True RUL', zorder=10)
ax.plot(pred_cycles, meta_pred, color='#c0392b', lw=3,
        label=f'WTA³ Ensemble  RMSE={meta_rmse:.2f}  MAE={meta_mae:.2f}  R²={meta_r2:.4f}',
        zorder=9)

for xv, lbl in [(111, 'Early|Mid'), (162, 'Mid|Late')]:
    ax.axvline(xv, color='gray', ls='--', lw=1.5, alpha=0.7)
    ax.text(xv+1, full_true_rul.max()*0.95, lbl, fontsize=9, color='gray')

ax.axvspan(1, CALIBRATION_CYCLES, alpha=0.08, color='#3498db',
           label=f'Calibration (cycles 1–{CALIBRATION_CYCLES})', zorder=0)
ax.axvline(CALIBRATION_CYCLES, color='#f39c12', ls='--', lw=2.5, alpha=0.9,
           zorder=8, label='Prediction Start (cycle 60)')

ax.set_xlabel('Cycle', fontsize=14, fontweight='bold')
ax.set_ylabel('RUL (cycles)', fontsize=14, fontweight='bold')
ax.set_title(f'Engine #52: WTA³ Meta-Ensemble RUL Prediction\n'
             f'({N_EXPERIMENTS} experiments × 6 models  |  '
             f'90% Variance CI  k90={k_90:.2f}σ)',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.25)
ax.set_xlim(0, TOTAL_CYCLES + 3)

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig2_main_rul_prediction.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig2")

# =============================================================================
# FIGURE 3 — Per-Model Mean Performance
# =============================================================================
print("Generating Figure 3...")

model_order = MODEL_NAMES + ['Meta-Ensemble']
rmse_vals   = [per_model_mean[m]['rmse'] for m in MODEL_NAMES] + [meta_rmse]
mae_vals    = [per_model_mean[m]['mae']  for m in MODEL_NAMES] + [meta_mae]
r2_vals     = [per_model_mean[m]['r2']   for m in MODEL_NAMES] + [meta_r2]
rmse_stds   = [float(np.std(per_model_test[m]['rmse'])) for m in MODEL_NAMES] + [0]
bar_colors  = colors6 + ['#2c3e50']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, vals, stds, ylabel, title in zip(
        axes,
        [rmse_vals, mae_vals, r2_vals],
        [rmse_stds, [0]*7, [0]*7],
        ['RMSE (cycles)', 'MAE (cycles)', 'R²'],
        ['RMSE', 'MAE', 'R²']):
    ax.bar(range(len(model_order)), vals, color=bar_colors,
           alpha=0.85, edgecolor='black', lw=1.5,
           yerr=stds, capsize=4, error_kw={'lw': 1.5})
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals)*0.015,
                f'{v:.3f}' if ylabel == 'R²' else f'{v:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel == 'R²':
        ax.set_ylim(0.85, 1.01)

plt.suptitle('Figure 3: Per-Model Performance — Test Engine #52\n'
             'Individual models: mean ± std across 100 experiments  |  '
             'Meta-Ensemble: mean of 100 WTA³ predictions',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig3_per_model_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig3")

# =============================================================================
# FIGURE 4 — Phase-wise MAE + Per-Model RMSE
# =============================================================================
print("Generating Figure 4...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

phases     = ['Early\n(61–111)', 'Mid\n(112–162)', 'Late\n(163–213)']
phase_maes = [meta_mae_e, meta_mae_m, meta_mae_l]
p_colors   = ['#f39c12', '#9b59b6', '#1abc9c']

for i, (v, c) in enumerate(zip(phase_maes, p_colors)):
    axes[0].bar(i, v, color=c, alpha=0.85, edgecolor='black', lw=1.5, width=0.5)
    axes[0].text(i, v + 0.05, f'{v:.2f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[0].set_xticks(range(3))
axes[0].set_xticklabels(phases, fontsize=11)
axes[0].set_ylabel('MAE (cycles)', fontsize=12, fontweight='bold')
axes[0].set_title('Meta-Ensemble Phase-wise MAE\n(Engine #52)',
                  fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

mean_rmses = [per_model_mean[m]['rmse'] for m in MODEL_NAMES]
std_rmses  = [float(np.std(per_model_test[m]['rmse'])) for m in MODEL_NAMES]
axes[1].bar(MODEL_NAMES, mean_rmses, color=colors6, alpha=0.85,
            edgecolor='black', lw=1.5, yerr=std_rmses, capsize=5,
            error_kw={'lw': 2})
axes[1].axhline(meta_rmse, color='#2c3e50', ls='--', lw=2,
                label=f'Meta-Ensemble RMSE={meta_rmse:.2f}')
for i, (v, s) in enumerate(zip(mean_rmses, std_rmses)):
    axes[1].text(i, v + s + 0.3, f'{v:.2f}±{s:.2f}',
                 ha='center', va='bottom', fontsize=8)
axes[1].set_ylabel('Mean RMSE (cycles)', fontsize=12, fontweight='bold')
axes[1].set_title('Per-Model Mean RMSE ± Std\nacross 100 Experiments',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('Figure 4: Phase-wise and Per-Model Performance Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig4_phase_analysis.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig4")

# =============================================================================
# FIGURE 5 — WTA³ Weight Distribution
# =============================================================================
print("Generating Figure 5...")

weights_per_model  = {m: [e['weights'][m]  for e in train_summary] for m in MODEL_NAMES}
outliers_per_model = {m: [e['outliers'][m] for e in train_summary] for m in MODEL_NAMES}
outlier_counts     = {m: sum(outliers_per_model[m]) for m in MODEL_NAMES}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, m in enumerate(MODEL_NAMES):
    ax = axes[idx]
    w  = weights_per_model[m]
    ax.hist(w, bins=20, color=colors6[idx], edgecolor='black', alpha=0.8, lw=1)
    ax.axvline(np.mean(w), color='black', lw=2, ls='--',
               label=f'Mean={np.mean(w):.3f}')
    ax.axvline(1/6, color='gray', lw=1.5, ls=':',
               label='Equal weight (1/6)')
    out_note = (f"\n[{outlier_counts[m]} configs: weight=0 (MAD excluded)]"
                if outlier_counts[m] > 0 else "")
    ax.set_title(f'{m}\nMean={np.mean(w):.3f}  Outliers={outlier_counts[m]}/100'
                 + out_note, fontsize=10, fontweight='bold')
    ax.set_xlabel('WTA³ Weight', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 5: WTA³ Weight Distribution per Model across 100 Experiments',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig5_weight_distribution.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig5")

# =============================================================================
# FIGURE 6 — Uncertainty Analysis (variance CI only)
# =============================================================================
print("Generating Figure 6...")

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

ax = axes[0]
ax.fill_between(pred_cycles, ens_preds.min(axis=0), ens_preds.max(axis=0),
                color='#bdc3c7', alpha=0.4, label='Full prediction range')
ax.fill_between(pred_cycles, ci_low_var, ci_high_var,
                color='#3498db', alpha=0.35,
                label=f'90% Variance CI  k90={k_90:.2f}σ  cov={cov_var:.0f}%')
ax.plot(full_cycles, full_true_rul, 'k-', lw=3, label='True RUL', zorder=10)
ax.plot(pred_cycles, meta_pred, color='#c0392b', lw=2.5,
        label='Meta-Ensemble', zorder=9)
ax.axvline(CALIBRATION_CYCLES, color='#f39c12', ls='--', lw=2.5,
           label='Prediction Start (cycle 60)')
for xv, lbl in [(111, 'Early|Mid'), (162, 'Mid|Late')]:
    ax.axvline(xv, color='gray', ls=':', lw=1.5)
    ax.text(xv+1, full_true_rul.max()*0.92, lbl, fontsize=9, color='gray')
ax.set_xlabel('Cycle', fontsize=12, fontweight='bold')
ax.set_ylabel('RUL (cycles)', fontsize=12, fontweight='bold')
ax.set_title(f'Variance Confidence Interval on Test Engine #52\n'
             f'(k90 calibrated on {n_cal_cycles} validation cycles — '
             f'test engine data not used in calibration)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.25)

ax2 = axes[1]
ax2.plot(pred_cycles, meta_std, color='#e74c3c', lw=2.5,
         label='Meta-Ensemble Std (epistemic)')
ax2.fill_between(pred_cycles, 0, meta_std, color='#e74c3c', alpha=0.2)
for xv, lbl in [(111, 'Early|Mid'), (162, 'Mid|Late')]:
    ax2.axvline(xv, color='gray', ls=':', lw=1.5)
    ax2.text(xv+1, meta_std.max()*0.95, lbl, fontsize=9, color='gray')
ax2.set_xlabel('Cycle', fontsize=12, fontweight='bold')
ax2.set_ylabel('Prediction Std (cycles)', fontsize=12, fontweight='bold')
ax2.set_title('Epistemic Uncertainty Evolution over Prediction Horizon',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.25)

plt.suptitle(f'Figure 6: Uncertainty Quantification — Variance-Based CI\n'
             f'k90={k_90:.2f}σ  |  Coverage={cov_var:.0f}%  |  '
             f'N_cal={n_cal_cycles} validation cycles',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig6_uncertainty_analysis.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig6")

# =============================================================================
# FIGURE 7 — Experiment Diversity
# =============================================================================
print("Generating Figure 7...")

exp_rmse_arr = np.array([e['ensemble_test']['rmse']      for e in eval_summary])
exp_nasa_arr = np.array([e['ensemble_test']['nasa_score'] for e in eval_summary])
exp_r2_arr   = np.array([e['ensemble_test']['r2']         for e in eval_summary])
val_rf_arr   = np.array([e['val_rmses']['RF']             for e in train_summary])
val_tr_arr   = np.array([e['val_rmses']['Transformer']    for e in train_summary])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc = axes[0].scatter(exp_rmse_arr, exp_nasa_arr, c=exp_r2_arr,
                     cmap='RdYlGn', s=60, alpha=0.8,
                     edgecolors='black', lw=0.5)
plt.colorbar(sc, ax=axes[0], label='R²')
axes[0].scatter([meta_rmse], [meta_nasa], color='black', s=200, marker='*',
                zorder=10, label=f'Meta-Ensemble\nRMSE={meta_rmse:.2f}')
best_rmse = agg_metrics['best_rmse']
best_nasa = next(e['ensemble_test']['nasa_score'] for e in eval_summary
                 if e['experiment'] == agg_metrics['best_experiment'])
axes[0].scatter([best_rmse], [best_nasa], color='red', s=200, marker='*',
                zorder=10,
                label=f'Best ({agg_metrics["best_experiment"]})\nRMSE={best_rmse:.2f}')
axes[0].set_xlabel('Ensemble RMSE (cycles)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('NASA Score', fontsize=12, fontweight='bold')
axes[0].set_title('RMSE vs NASA Score\n(100 experiments, colored by R²)',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

sc2 = axes[1].scatter(val_rf_arr, val_tr_arr, c=exp_rmse_arr,
                      cmap='RdYlGn_r', s=60, alpha=0.8,
                      edgecolors='black', lw=0.5)
plt.colorbar(sc2, ax=axes[1], label='Test RMSE')
axes[1].set_xlabel('Val RMSE — RF', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Val RMSE — Transformer', fontsize=12, fontweight='bold')
axes[1].set_title('RF vs Transformer Validation RMSE\n(colored by Test RMSE)',
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Figure 7: Experiment Diversity and Performance Landscape',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig7_diversity.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig7")

# =============================================================================
# FIGURE 8 — Prediction Horizon vs Absolute Error
# =============================================================================
print("Generating Figure 8...")

abs_err_meta = np.abs(meta_pred - true_rul)
horizon      = true_rul.copy()

fig, ax = plt.subplots(figsize=(14, 6))
ax2 = ax.twinx()

ax.plot(pred_cycles, horizon, color='#3498db', lw=3,
        label='Prediction Horizon (True RUL, left axis)')
ax.fill_between(pred_cycles, 0, horizon, color='#3498db', alpha=0.1)
ax2.plot(pred_cycles, abs_err_meta, color='#e74c3c', lw=2.5,
         label='Absolute Error (Meta-Ensemble, right axis)')
ax2.fill_between(pred_cycles, 0, abs_err_meta, color='#e74c3c', alpha=0.15)

z = np.polyfit(pred_cycles, abs_err_meta, 2)
ax2.plot(pred_cycles, np.poly1d(z)(pred_cycles), color='#c0392b',
         lw=2, ls='--', label='Error Trend')

for xv, lbl in [(111, 'Early|Mid'), (162, 'Mid|Late')]:
    ax.axvline(xv, color='gray', ls=':', lw=1.5)
    ax.text(xv+1, horizon.max()*0.92, lbl, fontsize=9, color='gray')

ax.set_xlabel('Cycle', fontsize=13, fontweight='bold')
ax.set_ylabel('Prediction Horizon (cycles)', fontsize=12,
              fontweight='bold', color='#3498db')
ax2.set_ylabel('Absolute Error (cycles)', fontsize=12,
               fontweight='bold', color='#e74c3c')
ax.tick_params(axis='y', labelcolor='#3498db')
ax2.tick_params(axis='y', labelcolor='#e74c3c')
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbl1+lbl2, fontsize=10, loc='upper right')
ax.set_title('Prediction Horizon vs Absolute Error\n'
             'Error decreases as failure approaches',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig8_horizon_error.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig8")

# =============================================================================
# FIGURE 9 — Overfitting Analysis
# =============================================================================
print("Generating Figure 9...")

mean_train = {m: float(np.mean([e['train_rmses'][m] for e in train_summary]))
              for m in MODEL_NAMES}
mean_val   = {m: float(np.mean([e['val_rmses'][m]   for e in train_summary]))
              for m in MODEL_NAMES}
gap_ratio  = {m: mean_val[m] / mean_train[m] for m in MODEL_NAMES}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(MODEL_NAMES))
w = 0.35

axes[0].bar(x - w/2, [mean_train[m] for m in MODEL_NAMES], w,
            label='Train RMSE', color='#3498db', alpha=0.8, edgecolor='black')
axes[0].bar(x + w/2, [mean_val[m]   for m in MODEL_NAMES], w,
            label='Val RMSE',   color='#e74c3c', alpha=0.8, edgecolor='black')
axes[0].set_xticks(x)
axes[0].set_xticklabels(MODEL_NAMES, fontsize=10)
axes[0].set_ylabel('Mean RMSE (cycles)', fontsize=12, fontweight='bold')
axes[0].set_title('Train vs Validation RMSE\n(Mean across 100 experiments)',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

gap_vals   = [gap_ratio[m] for m in MODEL_NAMES]
gap_colors = ['#27ae60' if g < 1.5 else '#f39c12' if g < 2.0 else '#e74c3c'
              for g in gap_vals]
axes[1].bar(MODEL_NAMES, gap_vals, color=gap_colors,
            alpha=0.85, edgecolor='black', lw=1.5)
axes[1].axhline(1.5, color='#27ae60', ls='--', lw=2, label='Good (<1.5×)')
axes[1].axhline(2.0, color='#f39c12', ls='--', lw=2, label='Warning (<2.0×)')
for i, v in enumerate(gap_vals):
    st = 'GOOD' if v < 1.5 else 'WARN' if v < 2.0 else 'OVERFIT'
    axes[1].text(i, v + 0.05, f'{v:.2f}×\n{st}',
                 ha='center', fontsize=9, fontweight='bold')
axes[1].set_ylabel('Gap Ratio (Val/Train RMSE)', fontsize=12, fontweight='bold')
axes[1].set_title('Overfitting Gap Ratio per Model', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('Figure 9: Overfitting Analysis — Train vs Validation Performance',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig9_overfitting.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig9")

# =============================================================================
# FIGURE 10 — Comprehensive Performance Table
# =============================================================================
print("Generating Figure 10...")

rows = []
for m in MODEL_NAMES:
    pm    = per_model_mean[m]
    std_r = float(np.std(per_model_test[m]['rmse']))
    std_a = float(np.std(per_model_test[m]['mae']))
    rows.append([m,
                 f"{pm['rmse']:.2f}±{std_r:.2f}",
                 f"{pm['mae']:.2f}±{std_a:.2f}",
                 f"{pm['r2']:.4f}",
                 '—', '—', '—'])

rows.append(['Meta-Ensemble',
             f"{meta_rmse:.2f}",
             f"{meta_mae:.2f}",
             f"{meta_r2:.4f}",
             f"{meta_mae_e:.2f}",
             f"{meta_mae_m:.2f}",
             f"{meta_mae_l:.2f}"])

col_labels = ['Model', 'RMSE (cycles)', 'MAE (cycles)', 'R²',
              'Early MAE\n(61–111)', 'Mid MAE\n(112–162)', 'Late MAE\n(163–213)']

fig, ax = plt.subplots(figsize=(22, 8))
ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=col_labels,
               cellLoc='center', loc='center',
               bbox=[0, 0.05, 1, 0.90])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.8)

for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#2c3e50')
    tbl[(0, j)].set_text_props(color='white', weight='bold')
for i in range(len(MODEL_NAMES)):
    for j in range(len(col_labels)):
        tbl[(i+1, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else '#ffffff')
for j in range(len(col_labels)):
    tbl[(len(MODEL_NAMES)+1, j)].set_facecolor('#aed6f1')
    tbl[(len(MODEL_NAMES)+1, j)].set_text_props(weight='bold')

ax.text(0.5, 0.99,
        'Figure 10: Comprehensive Performance Summary — WTA³ Meta-Ensemble Framework',
        ha='center', va='top', fontsize=14, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.02,
        'Individual model values: mean ± std across 100 independent experiments.  '
        'Meta-Ensemble: mean of 100 WTA³ weighted ensemble predictions.  '
        'All values are test-set performance on Engine #52.',
        ha='center', va='bottom', fontsize=9, style='italic',
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig10_performance_table.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig10")

# =============================================================================
# SENSITIVITY FIGURES — Tables + Line Plots
# =============================================================================
print("Generating Sensitivity Figures...")

# ── Sensitivity Line Plots only — tables go in the paper as LaTeX ─────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

beta_x    = [r['beta'] for r in beta_results]
beta_rmse = [r['rmse'] for r in beta_results]
beta_mae  = [r['mae']  for r in beta_results]
sel_beta_rmse = next(r['rmse'] for r in beta_results if r['selected'])

ax = axes[0]
ax.plot(beta_x, beta_rmse, 'o-', color='#e74c3c', lw=2.5, ms=8, label='RMSE', zorder=5)
ax.plot(beta_x, beta_mae,  's--', color='#3498db', lw=2,   ms=7, label='MAE',  zorder=5)
ax.scatter([-3], [sel_beta_rmse], color='#e74c3c', s=200, marker='*', zorder=10,
           label=f'Selected β=−3  RMSE={sel_beta_rmse:.4f}')
ax.set_xlabel('β (WTA³ exponent)', fontsize=12, fontweight='bold')
ax.set_ylabel('Error (cycles)', fontsize=12, fontweight='bold')
ax.set_title('β Exponent Sensitivity\n(MAD threshold = 2.5)',
             fontsize=13, fontweight='bold')
ax.set_xticks(beta_x)
ax.set_xticklabels([str(b) for b in beta_x], fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

mad_x    = [r['threshold']        for r in mad_results]
mad_rmse = [r['rmse']             for r in mad_results]
mad_mae  = [r['mae']              for r in mad_results]
mad_texc = [r['transformer_excl'] for r in mad_results]
sel_mad_rmse = next(r['rmse'] for r in mad_results if r['selected'])

ax2       = axes[1]
ax2_right = ax2.twinx()
ax2.plot(mad_x, mad_rmse, 'o-',  color='#e74c3c', lw=2.5, ms=8,
         label='RMSE (left)', zorder=5)
ax2.plot(mad_x, mad_mae,  's--', color='#3498db', lw=2,   ms=7,
         label='MAE (left)', zorder=5)
ax2_right.plot(mad_x, mad_texc, '^:', color='#9b59b6', lw=1.5, ms=7,
               label='Transformer excl/100 (right)', zorder=4)
ax2.scatter([2.5], [sel_mad_rmse], color='#e74c3c', s=200, marker='*', zorder=10,
            label=f'Selected 2.5  RMSE={sel_mad_rmse:.4f}')
ax2.set_xlabel('MAD Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error (cycles)', fontsize=12, fontweight='bold', color='#2c3e50')
ax2_right.set_ylabel('Transformer exclusions / 100',
                     fontsize=11, fontweight='bold', color='#9b59b6')
ax2_right.tick_params(axis='y', labelcolor='#9b59b6')
ax2.set_title('MAD Threshold Sensitivity\n(β = −3)',
              fontsize=13, fontweight='bold')
ax2.set_xticks(mad_x)
ax2.set_xticklabels([str(t) for t in mad_x], fontsize=11)
lines1, lbl1 = ax2.get_legend_handles_labels()
lines2, lbl2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines1+lines2, lbl1+lbl2, fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)

plt.suptitle('Sensitivity Analysis: Parameter Stability',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_sensitivity_lines.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_sensitivity_lines.png")

# =============================================================================
# SAVE FINAL CSV AND JSON
# =============================================================================
print("\nSaving final output files...")

df_meta = pd.DataFrame({
    'cycle'             : pred_cycles,
    'true_RUL'          : true_rul,
    'meta_ensemble_pred': meta_pred,
    'meta_ensemble_std' : meta_std,
    'CI_var_low'        : ci_low_var,
    'CI_var_high'       : ci_high_var,
    'abs_error'         : np.abs(meta_pred - true_rul),
})
df_meta.to_csv(os.path.join(RESULTS_DIR, 'meta_ensemble_predictions.csv'), index=False)

final_metrics = {
    'meta_ensemble': {
        'rmse'            : float(meta_rmse),
        'mae'             : float(meta_mae),
        'r2'              : float(meta_r2),
        'nasa_score'      : float(meta_nasa),
        'within_10_pct'   : float(meta_w10),
        'mae_early'       : float(meta_mae_e),
        'mae_mid'         : float(meta_mae_m),
        'mae_late'        : float(meta_mae_l),
        'nasa_early'      : float(meta_nasa_e),
        'nasa_mid'        : float(meta_nasa_m),
        'nasa_late'       : float(meta_nasa_l),
        'pred_start_cycle': PRED_START_CYCLE,
        'pred_end_cycle'  : TOTAL_CYCLES,
        'n_pred_points'   : n_pred,
    },
    'per_model_mean': {m: {k: float(v) for k, v in per_model_mean[m].items()}
                       for m in MODEL_NAMES},
    'uncertainty': {
        'k_90'                 : float(k_90),
        'variance_CI_coverage' : float(cov_var),
        'n_calibration_cycles' : n_cal_cycles,
        'calibration_source'   : 'validation_fleet_residuals_only',
        'note'                 : (
            'Conformal CI dropped. q90=118 cycles was uninformative '
            'due to fleet heterogeneity in validation residuals. '
            'Variance CI (k90=3.04σ) is the sole reported interval.'
        ),
    },
    'experiment_statistics': {
        'n_experiments'   : N_EXPERIMENTS,
        'rmse_mean'       : float(agg_metrics['ensemble_rmse_mean']),
        'rmse_std'        : float(agg_metrics['ensemble_rmse_std']),
        'best_experiment' : agg_metrics['best_experiment'],
        'best_rmse'       : float(agg_metrics['best_rmse']),
        'optimism_bias'   : optimism_bias_result,
    },
    'oracle_gap': oracle_summary,
}

with open(os.path.join(RESULTS_DIR, 'final_metrics.json'), 'w') as f:
    json.dump(final_metrics, f, indent=2)

print("  Saved: results/meta_ensemble_predictions.csv")
print("  Saved: results/final_metrics.json")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("  FINAL SUMMARY — WTA³ ENSEMBLE FRAMEWORK (v5)")
print("="*70)
print(f"\n  Meta-Ensemble Performance (Engine #52, cycles {PRED_START_CYCLE}–{TOTAL_CYCLES}):")
print(f"  RMSE        : {meta_rmse:.2f} cycles")
print(f"  MAE         : {meta_mae:.2f} cycles")
print(f"  R²          : {meta_r2:.4f}")
print(f"  NASA Score  : {meta_nasa:.4f}")
print(f"  Within ±10  : {meta_w10:.1f}%")
print(f"\n  Phase-wise MAE:")
print(f"  Early (61–111)  : {meta_mae_e:.2f}")
print(f"  Mid   (112–162) : {meta_mae_m:.2f}")
print(f"  Late  (163–213) : {meta_mae_l:.2f}")
print(f"\n  Uncertainty (variance CI only):")
print(f"  k90={k_90:.3f}  coverage={cov_var:.1f}%  N_cal={n_cal_cycles}")
print(f"  Conformal CI: DROPPED (fleet heterogeneity made q90 uninformative)")
print(f"\n  Optimism Bias: {bias_verdict}")
print(f"\n  All figures saved to: {FIGS_DIR}")
print("\n" + "="*70)
print("  STEP 6 v5 COMPLETE")
print("="*70)