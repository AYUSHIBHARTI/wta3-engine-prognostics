"""
extract_test_metrics.py
Extracts per-model mean test RMSE, MAE, R² on Engine #52
from evaluation_summary.json (already saved by Step 5).
Run this after Step 5 finishes.
"""

import json
import numpy as np

RESULTS_DIR = 'results'
MODEL_NAMES = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']

# ── Load evaluation summary ───────────────────────────────────────────────────
with open(f'{RESULTS_DIR}/evaluation_summary.json') as f:
    eval_summary = json.load(f)

with open(f'{RESULTS_DIR}/final_metrics.json') as f:
    final_metrics = json.load(f)

print(f"Loaded {len(eval_summary)} experiments\n")

# ── Per-model test metrics across 100 experiments ─────────────────────────────
# These are evaluated on Engine #52 test set — correct comparison
per_model = {m: {'rmse': [], 'mae': [], 'r2': []} for m in MODEL_NAMES}

for exp in eval_summary:
    for m in MODEL_NAMES:
        per_model[m]['rmse'].append(exp['per_model_test'][m]['rmse'])
        per_model[m]['mae'].append(exp['per_model_test'][m]['mae'])
        per_model[m]['r2'].append(exp['per_model_test'][m]['r2'])

# ── Compute mean ± std ────────────────────────────────────────────────────────
results = {}
for m in MODEL_NAMES:
    results[m] = {
        'rmse_mean': np.mean(per_model[m]['rmse']),
        'rmse_std':  np.std(per_model[m]['rmse']),
        'mae_mean':  np.mean(per_model[m]['mae']),
        'mae_std':   np.std(per_model[m]['mae']),
        'r2_mean':   np.mean(per_model[m]['r2']),
        'r2_std':    np.std(per_model[m]['r2']),
    }

# Meta-ensemble from final_metrics
meta = final_metrics['meta_ensemble']

# ── Print the comparison table ────────────────────────────────────────────────
print("=" * 75)
print("  PERFORMANCE TABLE — Test Engine #52 (mean across 100 experiments)")
print("=" * 75)
print(f"{'Model':<14} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("-" * 75)

for m in MODEL_NAMES:
    r = results[m]
    print(f"{m:<14} "
          f"{r['rmse_mean']:>6.2f}±{r['rmse_std']:.2f}  "
          f"{r['mae_mean']:>6.2f}±{r['mae_std']:.2f}  "
          f"{r['r2_mean']:>6.4f}±{r['r2_std']:.4f}")

print("-" * 75)
print(f"{'Meta-Ensemble':<14} "
      f"{meta['rmse']:>10.2f}  "
      f"{meta['mae']:>10.2f}  "
      f"{meta['r2']:>10.4f}")
print("=" * 75)

# ── Verify hypothesis ─────────────────────────────────────────────────────────
print("\nHypothesis Check: Meta-Ensemble vs Individual Models")
print("-" * 50)
meta_rmse = meta['rmse']
meta_mae  = meta['mae']
meta_r2   = meta['r2']

all_beat_rmse = True
all_beat_mae  = True
all_beat_r2   = True

for m in MODEL_NAMES:
    r = results[m]
    beat_rmse = meta_rmse < r['rmse_mean']
    beat_mae  = meta_mae  < r['mae_mean']
    beat_r2   = meta_r2   > r['r2_mean']

    rmse_imp = (r['rmse_mean'] - meta_rmse) / r['rmse_mean'] * 100
    print(f"  vs {m:<12}: RMSE {'✓' if beat_rmse else '✗'} "
          f"({r['rmse_mean']:.2f}→{meta_rmse:.2f}, {rmse_imp:+.1f}%)  "
          f"MAE {'✓' if beat_mae else '✗'}  "
          f"R² {'✓' if beat_r2 else '✗'}")

    if not beat_rmse: all_beat_rmse = False
    if not beat_mae:  all_beat_mae  = False
    if not beat_r2:   all_beat_r2   = False

print()
if all_beat_rmse and all_beat_mae and all_beat_r2:
    print("  ✓ HYPOTHESIS CONFIRMED: Meta-Ensemble outperforms ALL individual models")
    best_ind_rmse = max(results[m]['rmse_mean'] for m in MODEL_NAMES)
    worst_ind_rmse = max(results[m]['rmse_mean'] for m in MODEL_NAMES)
    best_ind = min(MODEL_NAMES, key=lambda m: results[m]['rmse_mean'])
    improvement = (results[best_ind]['rmse_mean'] - meta_rmse) / results[best_ind]['rmse_mean'] * 100
    print(f"  Best individual model: {best_ind} (RMSE={results[best_ind]['rmse_mean']:.2f})")
    print(f"  Improvement over best individual: {improvement:.1f}% RMSE reduction")
else:
    print("  ✗ Partial: Meta-Ensemble does not beat all models on all metrics")

# ── Save to JSON for thesis use ───────────────────────────────────────────────
output = {
    'individual_models_test': {
        m: {k: float(v) for k, v in results[m].items()}
        for m in MODEL_NAMES
    },
    'meta_ensemble_test': {
        'rmse': float(meta_rmse),
        'mae':  float(meta_mae),
        'r2':   float(meta_r2),
    }
}

with open(f'{RESULTS_DIR}/thesis_comparison_table.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: results/thesis_comparison_table.json")