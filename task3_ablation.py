# =============================================================================
# task3_ablation.py
# Ablation Study — 4 conditions compared against WTA³ (full framework)
#
# Conditions:
#   A. Simple Average       — equal weights, no MAD filter
#   B. No MAD Filter        — WTA³ weights but no outlier removal
#   C. Single Best Model    — best val-RMSE model per experiment, weight=1
#   D. WTA³ No DL           — WTA³ weights on RF + XGB + SVR only
#   E. WTA³ Full (baseline) — β=−3, MAD=2.5, all 6 models  ← your paper result
#
# Output:
#   results/ablation_results.json   — full numbers
#   results/figures/fig_ablation.png — bar chart for paper
# =============================================================================

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import median_abs_deviation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — must match step6_v5 exactly
# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION_CYCLES = 60
TOTAL_CYCLES       = 213
N_EXPERIMENTS      = 100
MODEL_NAMES        = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
ML_ONLY            = ['RF', 'XGB', 'SVR']
DL_ONLY            = ['LSTM', 'CNN', 'Transformer']
RESULTS_DIR        = 'results'
FIGS_DIR           = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

PRED_START_CYCLE = CALIBRATION_CYCLES + 1
N_PRED           = TOTAL_CYCLES - PRED_START_CYCLE + 1   # 153

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading saved results...")

with open(os.path.join(RESULTS_DIR, 'all_predictions.json'))  as f: all_preds     = json.load(f)
with open(os.path.join(RESULTS_DIR, 'training_summary.json')) as f: train_summary = json.load(f)

# Ground truth
true_rul = np.arange(TOTAL_CYCLES - PRED_START_CYCLE, -1, -1)  # (153,)

# Per-model predictions: shape (100, 153)
all_model_preds = {m: [] for m in MODEL_NAMES}
for exp in all_preds:
    for m in MODEL_NAMES:
        all_model_preds[m].append(np.array(exp['test_predictions'][m]))
for m in MODEL_NAMES:
    all_model_preds[m] = np.array(all_model_preds[m])

# Val RMSEs per experiment per model
all_val_rmses = [{m: exp['val_rmses'][m] for m in MODEL_NAMES}
                 for exp in train_summary]

print(f"  Loaded {N_EXPERIMENTS} experiments  |  N_PRED={N_PRED}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred, true):
    rmse  = float(np.sqrt(mean_squared_error(true, pred)))
    mae   = float(mean_absolute_error(true, pred))
    r2    = float(r2_score(true, pred))
    e     = np.array(pred) - np.array(true)
    nasa  = float(np.sum(np.where(e < 0,
                                  np.exp(-e / 13.0) - 1,
                                  np.exp( e / 10.0) - 1)))
    w10   = float(np.mean(np.abs(true - pred) <= 10) * 100)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'nasa': nasa, 'within_10': w10}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — WTA³ weight calculation (reused from step6)
# ─────────────────────────────────────────────────────────────────────────────
def wta3_weights(val_rmses_dict, model_subset,
                 alpha=0.05, beta=-3, mad_threshold=2.5, apply_mad=True):
    vals     = np.array([val_rmses_dict[m] for m in model_subset], dtype=float)
    median_r = np.median(vals)
    mad      = float(median_abs_deviation(vals))

    if apply_mad and mad > 1e-10:
        outlier = np.abs(vals - median_r) > mad_threshold * mad
    else:
        outlier = np.zeros(len(vals), dtype=bool)

    inlier_vals = vals[~outlier]
    mean_inlier = float(np.mean(inlier_vals)) if len(inlier_vals) > 0 \
                  else float(np.mean(vals))

    raw = np.zeros(len(vals))
    for i, r in enumerate(vals):
        if not outlier[i]:
            raw[i] = (r + alpha * mean_inlier) ** beta

    total = raw.sum()
    if total > 0:
        raw /= total
    else:
        n_in = (~outlier).sum()
        raw  = np.where(~outlier, 1.0 / n_in if n_in > 0 else 0.0, 0.0)

    return {m: float(raw[i]) for i, m in enumerate(model_subset)}

# =============================================================================
# CONDITION A — Simple Average (equal weights, no MAD, all 6 models)
# =============================================================================
print("\nCondition A: Simple Average...")

cond_a_preds = []
for exp_idx in range(N_EXPERIMENTS):
    ens = np.mean([all_model_preds[m][exp_idx] for m in MODEL_NAMES], axis=0)
    cond_a_preds.append(ens)

meta_a = np.array(cond_a_preds).mean(axis=0)
metrics_a = compute_metrics(meta_a, true_rul)
print(f"  RMSE={metrics_a['rmse']:.4f}  MAE={metrics_a['mae']:.4f}  R²={metrics_a['r2']:.4f}")

# =============================================================================
# CONDITION B — WTA³ weights but NO MAD filter
# =============================================================================
print("Condition B: WTA³ No MAD Filter...")

cond_b_preds = []
for exp_idx in range(N_EXPERIMENTS):
    w   = wta3_weights(all_val_rmses[exp_idx], MODEL_NAMES,
                       apply_mad=False, beta=-3)
    ens = sum(w[m] * all_model_preds[m][exp_idx] for m in MODEL_NAMES)
    cond_b_preds.append(ens)

meta_b = np.array(cond_b_preds).mean(axis=0)
metrics_b = compute_metrics(meta_b, true_rul)
print(f"  RMSE={metrics_b['rmse']:.4f}  MAE={metrics_b['mae']:.4f}  R²={metrics_b['r2']:.4f}")

# =============================================================================
# CONDITION C — Single Best Model per experiment (winner-takes-all)
# Best = lowest val RMSE across all 6 models in that experiment
# =============================================================================
print("Condition C: Single Best Model (per experiment)...")

cond_c_preds = []
best_model_counts = {m: 0 for m in MODEL_NAMES}

for exp_idx in range(N_EXPERIMENTS):
    val_r    = all_val_rmses[exp_idx]
    best_m   = min(val_r, key=val_r.get)
    best_model_counts[best_m] += 1
    cond_c_preds.append(all_model_preds[best_m][exp_idx])

meta_c = np.array(cond_c_preds).mean(axis=0)
metrics_c = compute_metrics(meta_c, true_rul)
print(f"  RMSE={metrics_c['rmse']:.4f}  MAE={metrics_c['mae']:.4f}  R²={metrics_c['r2']:.4f}")
print(f"  Best model selection counts: {best_model_counts}")

# =============================================================================
# CONDITION D — WTA³ ML-only (RF + XGB + SVR, no deep learning)
# =============================================================================
print("Condition D: WTA³ ML-Only (RF + XGB + SVR)...")

cond_d_preds = []
for exp_idx in range(N_EXPERIMENTS):
    w   = wta3_weights(all_val_rmses[exp_idx], ML_ONLY,
                       apply_mad=True, beta=-3, mad_threshold=2.5)
    ens = sum(w[m] * all_model_preds[m][exp_idx] for m in ML_ONLY)
    cond_d_preds.append(ens)

meta_d = np.array(cond_d_preds).mean(axis=0)
metrics_d = compute_metrics(meta_d, true_rul)
print(f"  RMSE={metrics_d['rmse']:.4f}  MAE={metrics_d['mae']:.4f}  R²={metrics_d['r2']:.4f}")

# =============================================================================
# CONDITION E — WTA³ Full (your paper result — baseline)
# =============================================================================
print("Condition E: WTA³ Full (β=−3, MAD=2.5, all 6 models)...")

cond_e_preds = np.array([e['ensemble_test_pred'] for e in all_preds])
meta_e       = cond_e_preds.mean(axis=0)
metrics_e    = compute_metrics(meta_e, true_rul)
print(f"  RMSE={metrics_e['rmse']:.4f}  MAE={metrics_e['mae']:.4f}  R²={metrics_e['r2']:.4f}")

# =============================================================================
# PER-EXPERIMENT RMSE — std across 100 experiments per condition
# This is the robustness argument: lower std = more reliable across configs
# =============================================================================
print("\nComputing per-experiment RMSE std across 100 experiments...")

def per_exp_rmse_list(exp_preds_list):
    """exp_preds_list: list of 100 arrays, each (N_PRED,)"""
    return [float(np.sqrt(mean_squared_error(true_rul, p)))
            for p in exp_preds_list]

per_exp_a = per_exp_rmse_list(cond_a_preds)
per_exp_b = per_exp_rmse_list(cond_b_preds)
per_exp_c = per_exp_rmse_list(cond_c_preds)
per_exp_d = per_exp_rmse_list(cond_d_preds)
per_exp_e = [float(np.sqrt(mean_squared_error(true_rul, p)))
             for p in cond_e_preds]

std_a = float(np.std(per_exp_a))
std_b = float(np.std(per_exp_b))
std_c = float(np.std(per_exp_c))
std_d = float(np.std(per_exp_d))
std_e = float(np.std(per_exp_e))

p90_a = float(np.percentile(per_exp_a, 90))
p90_b = float(np.percentile(per_exp_b, 90))
p90_c = float(np.percentile(per_exp_c, 90))
p90_d = float(np.percentile(per_exp_d, 90))
p90_e = float(np.percentile(per_exp_e, 90))

worst_a = float(np.max(per_exp_a))
worst_b = float(np.max(per_exp_b))
worst_c = float(np.max(per_exp_c))
worst_d = float(np.max(per_exp_d))
worst_e = float(np.max(per_exp_e))

# =============================================================================
# COMPILE RESULTS — now includes std, p90, worst-case
# =============================================================================
conditions = [
    ('Simple Average',    'Equal weights, no MAD filter',           metrics_a, std_a, p90_a, worst_a, per_exp_a),
    ('WTA³ No MAD',       'β=−3 weights, MAD filter disabled',      metrics_b, std_b, p90_b, worst_b, per_exp_b),
    ('Single Best Model', 'Winner-takes-all per experiment',         metrics_c, std_c, p90_c, worst_c, per_exp_c),
    ('WTA³ ML-Only',      'RF+XGB+SVR only, β=−3, MAD=2.5',        metrics_d, std_d, p90_d, worst_d, per_exp_d),
    ('WTA³ Full',         'All 6 models, β=−3, MAD=2.5 (proposed)', metrics_e, std_e, p90_e, worst_e, per_exp_e),
]

print("\n" + "="*90)
print(f"  {'Condition':<22}  {'RMSE':>7}  {'MAE':>7}  {'R²':>8}  "
      f"{'W10%':>6}  {'Std':>7}  {'P90':>7}  {'Worst':>7}")
print("  " + "-"*88)
for name, desc, m, std, p90, worst, _ in conditions:
    marker = "  ← PROPOSED" if name == 'WTA³ Full' else ""
    print(f"  {name:<22}  {m['rmse']:>7.4f}  {m['mae']:>7.4f}  "
          f"{m['r2']:>8.4f}  {m['within_10']:>6.1f}%  "
          f"{std:>7.4f}  {p90:>7.4f}  {worst:>7.4f}{marker}")
print("="*90)

ablation_results = {
    'conditions': [
        {
            'name'            : name,
            'description'     : desc,
            'metrics'         : m,
            'per_exp_rmse_std': std,
            'per_exp_rmse_p90': p90,
            'per_exp_rmse_worst': worst,
        }
        for name, desc, m, std, p90, worst, _ in conditions
    ],
    'best_model_counts': best_model_counts,
    'note': (
        'Condition E (WTA³ Full) is the proposed method. '
        'All conditions evaluated on Engine #52, cycles 61–213. '
        'Meta-prediction = mean of 100 experiment ensemble predictions. '
        'Std/P90/Worst are across 100 per-experiment RMSE values — '
        'lower = more robust across hyperparameter configurations.'
    ),
}

with open(os.path.join(RESULTS_DIR, 'ablation_results.json'), 'w') as f:
    json.dump(ablation_results, f, indent=2)
print(f"\n  Saved: results/ablation_results.json")

# =============================================================================
# FIGURE — Ablation: 2-panel (Meta RMSE + Per-Experiment Std)
# The std panel is the robustness argument
# =============================================================================
print("\nGenerating ablation figure...")

labels      = [c[0] for c in conditions]
rmse_vals   = [c[2]['rmse'] for c in conditions]
std_vals    = [c[3]         for c in conditions]
worst_vals  = [c[5]         for c in conditions]
w10_vals    = [c[2]['within_10'] for c in conditions]
per_exp_all = [c[6]         for c in conditions]

bar_colors  = ['#95a5a6', '#95a5a6', '#95a5a6', '#95a5a6', '#2c3e50']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Panel 1: Meta RMSE ────────────────────────────────────────────────────────
ax = axes[0]
ref_rmse = rmse_vals[-1]
bars = ax.bar(range(len(labels)), rmse_vals,
              color=bar_colors, edgecolor='black', alpha=0.85, lw=1.5)
ax.axhline(ref_rmse, color='#2c3e50', ls='--', lw=1.5, alpha=0.5)
for i, v in enumerate(rmse_vals):
    delta = v - ref_rmse
    sign  = '+' if delta > 0 else ''
    label_txt = f'{v:.2f}' if i == len(labels)-1 else f'{v:.2f}\n({sign}{delta:.2f})'
    color_txt = '#2c3e50' if i == len(labels)-1 else 'black'
    ax.text(i, v + 0.05, label_txt,
            ha='center', va='bottom', fontsize=8,
            fontweight='bold', color=color_txt)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Meta-Ensemble RMSE (cycles)', fontsize=12, fontweight='bold')
ax.set_title('Mean RMSE across 100 Experiments\n(lower = better average accuracy)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Panel 2: Per-Experiment RMSE std + worst-case ────────────────────────────
ax2 = axes[1]
ref_std = std_vals[-1]
bars2 = ax2.bar(range(len(labels)), std_vals,
                color=bar_colors, edgecolor='black', alpha=0.85, lw=1.5)
ax2.axhline(ref_std, color='#2c3e50', ls='--', lw=1.5, alpha=0.5)

# Overlay worst-case as scatter
ax2_r = ax2.twinx()
ax2_r.scatter(range(len(labels)), worst_vals,
              color=['#e74c3c' if i < len(labels)-1 else '#c0392b'
                     for i in range(len(labels))],
              s=80, zorder=5, marker='D',
              label='Worst-case RMSE (right axis)')
ax2_r.set_ylabel('Worst-case RMSE (cycles)', fontsize=11,
                 fontweight='bold', color='#e74c3c')
ax2_r.tick_params(axis='y', labelcolor='#e74c3c')

for i, v in enumerate(std_vals):
    delta = v - ref_std
    sign  = '+' if delta > 0 else ''
    label_txt = f'{v:.2f}' if i == len(labels)-1 else f'{v:.2f}\n({sign}{delta:.2f})'
    color_txt = '#2c3e50' if i == len(labels)-1 else 'black'
    ax2.text(i, v + 0.02, label_txt,
             ha='center', va='bottom', fontsize=8,
             fontweight='bold', color=color_txt)

ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax2.set_ylabel('Per-Experiment RMSE Std (cycles)', fontsize=12, fontweight='bold')
ax2.set_title('Robustness across 100 Configurations\n'
              '(lower std = more stable across hyperparameter diversity)',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.spines['top'].set_visible(False)

lines, lbls = ax2_r.get_legend_handles_labels()
ax2.legend(lines, lbls, fontsize=9, loc='upper left')

plt.suptitle('Figure: Ablation Study — WTA³ Component Contributions\n'
             'Engine #52, cycles 61–213  |  '
             'Dark bar = proposed WTA³ Full method',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_ablation.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/fig_ablation.png")

# =============================================================================
# PAPER-READY SUMMARY
# =============================================================================
print("\n" + "="*70)
print("  ABLATION SUMMARY — PAPER-READY NUMBERS")
print("="*70)
wta3_rmse = metrics_e['rmse']
wta3_mae  = metrics_e['mae']
wta3_std  = std_e
wta3_worst = worst_e

print(f"\n  WTA³ Full (proposed): RMSE={wta3_rmse:.4f}  "
      f"Std={wta3_std:.4f}  Worst={wta3_worst:.4f}")
print()

for name, desc, m, std, p90, worst, _ in conditions[:-1]:
    rmse_delta = m['rmse'] - wta3_rmse
    std_delta  = std - wta3_std
    worst_delta = worst - wta3_worst
    print(f"  vs {name}:")
    print(f"    Mean RMSE  : {m['rmse']:.4f}  (WTA³ Full {rmse_delta:+.4f} cycles)")
    print(f"    RMSE Std   : {std:.4f}        (WTA³ Full {std_delta:+.4f} — "
          + ("MORE stable)" if std_delta < 0 else "LESS stable)"))
    print(f"    Worst-case : {worst:.4f}       (WTA³ Full {worst_delta:+.4f})")
    print()