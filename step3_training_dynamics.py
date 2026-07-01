# =============================================================================
# step3_training_dynamics.py
# Analyse overfitting gap (val_rmse / train_rmse) and val RMSE z-score
# as candidate single-model reliability signals.
# Key question: can we detect Type1/Type2 outliers from training history alone?
# =============================================================================

import json
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve

plt.style.use('default')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

RESULTS_DIR = 'results'
STEP1_DIR   = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step1')
OUT_DIR     = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step3')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
COLORS      = {'Type1': '#D32F2F', 'Type2': '#1976D2', 'Inlier': '#388E3C'}
MCOLORS     = ['#4FC3F7','#81C784','#FFB74D','#F06292','#CE93D8','#80DEEA']

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
with open(os.path.join(RESULTS_DIR, 'training_summary.json')) as f:
    training = json.load(f)

records = []
with open(os.path.join(STEP1_DIR, 'outlier_type_labels.csv')) as f:
    for row in csv.DictReader(f):
        row['val_rmse']   = float(row['val_rmse'])
        row['train_rmse'] = float(row['train_rmse'])
        row['gap_ratio']  = float(row['gap_ratio'])
        row['outlier']    = row['outlier'] == 'True'
        records.append(row)

print(f"  {len(records)} (experiment x model) records loaded\n")

# ── Compute z-score of val RMSE within each model type ───────────────────────
# This normalises across model types so we can compare apples to apples
for m in MODEL_NAMES:
    m_recs = [r for r in records if r['model'] == m]
    vals   = np.array([r['val_rmse'] for r in m_recs])
    mu, sd = vals.mean(), vals.std()
    for r in m_recs:
        r['val_rmse_zscore'] = (r['val_rmse'] - mu) / (sd + 1e-8)

# Also z-score the gap ratio within each model type
for m in MODEL_NAMES:
    m_recs = [r for r in records if r['model'] == m]
    gaps   = np.array([r['gap_ratio'] for r in m_recs])
    mu, sd = gaps.mean(), gaps.std()
    for r in m_recs:
        r['gap_zscore'] = (r['gap_ratio'] - mu) / (sd + 1e-8)

# ── Print summary statistics ──────────────────────────────────────────────────
print("=" * 65)
print("  GAP RATIO & VAL RMSE Z-SCORE BY OUTLIER TYPE")
print("=" * 65)
print(f"\n{'Type':<10} {'Gap mean':>10} {'Gap std':>9} {'Gap median':>11} "
      f"{'ValZ mean':>10} {'ValZ std':>9} {'N':>5}")
print("-" * 70)
for otype in ['Inlier', 'Type1', 'Type2']:
    recs  = [r for r in records if r['otype'] == otype]
    gaps  = [r['gap_ratio']      for r in recs]
    valzs = [r['val_rmse_zscore'] for r in recs]
    print(f"  {otype:<8} {np.mean(gaps):>10.3f} {np.std(gaps):>9.3f} "
          f"{np.median(gaps):>11.3f} {np.mean(valzs):>10.3f} "
          f"{np.std(valzs):>9.3f} {len(recs):>5}")

print()
print(f"{'Model':<14} {'Type':<10} {'Gap mean':>10} {'Gap std':>9} "
      f"{'ValRMSE mean':>13} {'ValZ mean':>10}")
print("-" * 72)
for m in MODEL_NAMES:
    for otype in ['Inlier', 'Type1', 'Type2']:
        recs = [r for r in records if r['model'] == m and r['otype'] == otype]
        if not recs:
            continue
        gaps  = [r['gap_ratio']       for r in recs]
        vals  = [r['val_rmse']        for r in recs]
        valzs = [r['val_rmse_zscore'] for r in recs]
        print(f"  {m:<12} {otype:<10} {np.mean(gaps):>10.3f} {np.std(gaps):>9.3f} "
              f"{np.mean(vals):>13.2f} {np.mean(valzs):>10.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Gap ratio distributions: Inlier vs Type1 vs Type2
# ═════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Gap ratio distributions...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: histogram (clip at 10 for readability)
ax = axes[0]
for otype, color in [('Inlier','#388E3C'),('Type1','#D32F2F'),('Type2','#1976D2')]:
    gaps = np.clip([r['gap_ratio'] for r in records if r['otype'] == otype], 0, 8)
    ax.hist(gaps, bins=35, alpha=0.6, color=color, density=True,
            label=f"{otype}  mean={np.mean(gaps):.2f}")
ax.axvline(1.0, color='black', linestyle='--', linewidth=1.2, label='Gap = 1.0')
ax.set_xlabel('Gap Ratio  (val / train RMSE)  [clipped at 8]', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Gap Ratio Distribution by Type\n(all models combined)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# Right: violin per model, coloured by type
ax = axes[1]
positions = []
violin_data = []
violin_colors = []
tick_labels = []
pos = 1
for m in MODEL_NAMES:
    for otype, color in [('Inlier','#388E3C'),('Type1','#D32F2F'),('Type2','#1976D2')]:
        recs = [r for r in records if r['model'] == m and r['otype'] == otype]
        if len(recs) < 2:
            continue
        gaps = np.clip([r['gap_ratio'] for r in recs], 0, 8)
        violin_data.append(gaps)
        violin_colors.append(color)
        positions.append(pos)
        tick_labels.append(f'{m}\n{otype[:3]}')
        pos += 1
    pos += 0.5

vp = ax.violinplot(violin_data, positions=positions, showmedians=True,
                   showextrema=True)
for body, color in zip(vp['bodies'], violin_colors):
    body.set_facecolor(color)
    body.set_alpha(0.6)
for part in ['cmedians','cmaxes','cmins','cbars']:
    vp[part].set_color('black')
    vp[part].set_linewidth(0.8)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
ax.set_xticks(positions)
ax.set_xticklabels(tick_labels, fontsize=6.5)
ax.set_ylabel('Gap Ratio (clipped at 8)', fontsize=10)
ax.set_title('Gap Ratio by Model × Type', fontsize=11)
ax.grid(axis='y', alpha=0.25)

patches = [mpatches.Patch(color='#388E3C', label='Inlier'),
           mpatches.Patch(color='#D32F2F', label='Type 1'),
           mpatches.Patch(color='#1976D2', label='Type 2')]
ax.legend(handles=patches, fontsize=8)

fig.suptitle('Step 3 — Overfitting Gap: Can Training History Predict Outlier Type?',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's3_fig1_gap_ratio_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s3_fig1_gap_ratio_distributions.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Val RMSE z-score by type per model
# This is the normalised version — compares models fairly
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 2: Val RMSE z-score per model...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax   = axes[j]
    data = []
    lbls = []
    cols = []
    for otype, color in [('Inlier','#388E3C'),('Type1','#D32F2F'),('Type2','#1976D2')]:
        recs = [r for r in records if r['model'] == m and r['otype'] == otype]
        if recs:
            data.append([r['val_rmse_zscore'] for r in recs])
            lbls.append(f'{otype}\n(n={len(recs)})')
            cols.append(color)

    bp = ax.boxplot(data, labels=lbls, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], cols):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(2, color='orange', linestyle=':', linewidth=1, label='+2σ')
    ax.axhline(-2, color='blue',  linestyle=':', linewidth=1, label='-2σ')
    ax.set_title(m, fontsize=11)
    ax.set_ylabel('Val RMSE Z-score', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.2)

fig.suptitle('Step 3 — Val RMSE Z-score by Model and Outlier Type\n'
             '(z-score normalised within each model type — 0 = average, ±2 = unusual)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's3_fig2_valrmse_zscore_per_model.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s3_fig2_valrmse_zscore_per_model.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Gap ratio vs val RMSE scatter (all models, coloured by type)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 3: Gap ratio vs val RMSE scatter...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax   = axes[j]
    recs = [r for r in records if r['model'] == m]
    xs   = np.clip([r['gap_ratio'] for r in recs], 0, 8)
    ys   = [r['val_rmse'] for r in recs]
    cols = [COLORS[r['otype']] for r in recs]
    ax.scatter(xs, ys, c=cols, s=40, alpha=0.8, zorder=2)
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.8)

    # Annotate Type1 points
    for r, x, y in zip(recs, xs, ys):
        if r['otype'] == 'Type1':
            ax.annotate(r['experiment'][-3:], (x, y),
                        textcoords='offset points', xytext=(4, 3),
                        fontsize=6, color='#D32F2F')

    ax.set_title(m, fontsize=11)
    ax.set_xlabel('Gap Ratio (clipped at 8)', fontsize=8)
    ax.set_ylabel('Val RMSE', fontsize=8)
    ax.grid(alpha=0.2)

patches = [mpatches.Patch(color=COLORS[t], label=t)
           for t in ['Inlier','Type1','Type2']]
fig.legend(handles=patches, loc='lower center', ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Step 3 — Gap Ratio vs Val RMSE per Model\n'
             'Do outlier types occupy distinct regions of this 2D space?',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's3_fig3_gap_vs_valrmse_scatter.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s3_fig3_gap_vs_valrmse_scatter.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — ROC curves: how well do gap ratio and val RMSE z-score
#            detect Type1 and Type2 outliers individually?
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 4: ROC curves for gap ratio and val RMSE z-score...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, target, title in zip(
        axes,
        ['Type1', 'Type2'],
        ['Detecting Type 1 (Bad — high RMSE)', 'Detecting Type 2 (Divergent — low RMSE)']):

    y_true = np.array([1 if r['otype'] == target else 0 for r in records])

    features = {
        'Val RMSE z-score'      : np.array([r['val_rmse_zscore'] for r in records]),
        'Gap ratio'             : np.array([r['gap_ratio']        for r in records]),
        'Gap z-score'           : np.array([r['gap_zscore']       for r in records]),
        'Val RMSE (raw)'        : np.array([r['val_rmse']         for r in records]),
    }

    # For Type2, low val RMSE is the signal -> invert zscore
    if target == 'Type2':
        features['Val RMSE z-score (inv)'] = -features['Val RMSE z-score']
        features.pop('Val RMSE z-score')

    line_styles = ['-', '--', '-.', ':']
    for (fname, fvals), ls in zip(features.items(), line_styles):
        fpr, tpr, _ = roc_curve(y_true, fvals)
        roc_auc     = auc(fpr, tpr)
        # flip if AUC < 0.5
        if roc_auc < 0.5:
            fpr, tpr, _ = roc_curve(y_true, -fvals)
            roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, ls, linewidth=2,
                label=f'{fname}  AUC={roc_auc:.3f}')

    ax.plot([0,1],[0,1], 'k--', linewidth=0.8, alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)

fig.suptitle('Step 3 — ROC Curves: Training Signals as Single-Model Outlier Detectors\n'
             '(AUC = 0.5 is random; AUC = 1.0 is perfect)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's3_fig4_roc_curves.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s3_fig4_roc_curves.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Gap ratio vs test MAE: does a high gap predict bad test perf?
# Load predictions for test error
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5: Gap ratio vs test MAE...")
with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json')) as f:
    eval_summary = json.load(f)

# Build map: (experiment, model) -> test MAE
test_mae_map = {}
for row in eval_summary:
    exp = row['experiment']
    for m in MODEL_NAMES:
        test_mae_map[(exp, m)] = row['per_model_test'][m]['mae']

for r in records:
    r['test_mae'] = test_mae_map.get((r['experiment'], r['model']), np.nan)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax   = axes[j]
    recs = [r for r in records if r['model'] == m and not np.isnan(r['test_mae'])]
    xs   = np.clip([r['gap_ratio'] for r in recs], 0, 8)
    ys   = [r['test_mae']  for r in recs]
    cols = [COLORS[r['otype']] for r in recs]

    ax.scatter(xs, ys, c=cols, s=40, alpha=0.8, zorder=2)

    # Regression line for inliers
    in_x = np.clip([r['gap_ratio'] for r in recs if r['otype']=='Inlier'], 0, 8)
    in_y = [r['test_mae']  for r in recs if r['otype']=='Inlier']
    if len(in_x) > 2:
        coef = np.polyfit(in_x, in_y, 1)
        xline = np.linspace(min(in_x), max(in_x), 50)
        ax.plot(xline, np.polyval(coef, xline), color='gray',
                linewidth=1, linestyle='--', alpha=0.7, label='Inlier trend')

    ax.set_title(m, fontsize=11)
    ax.set_xlabel('Gap Ratio (clipped at 8)', fontsize=8)
    ax.set_ylabel('Test MAE (cycles)', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

patches = [mpatches.Patch(color=COLORS[t], label=t)
           for t in ['Inlier','Type1','Type2']]
fig.legend(handles=patches, loc='lower center', ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Step 3 — Gap Ratio vs Test MAE\n'
             'Does overfitting gap predict actual test error? (grey line = inlier trend)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's3_fig5_gap_vs_test_mae.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s3_fig5_gap_vs_test_mae.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Threshold analysis: precision & recall vs threshold
#            For the best single feature per outlier type
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 6: Precision-recall vs threshold...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, target, feature_key, invert, title in [
    (axes[0], 'Type1', 'val_rmse_zscore', False,
     'Type 1 detector  (feature: Val RMSE z-score)'),
    (axes[1], 'Type2', 'val_rmse_zscore', True,
     'Type 2 detector  (feature: -Val RMSE z-score)')]:

    y_true = np.array([1 if r['otype'] == target else 0 for r in records])
    scores = np.array([r[feature_key] for r in records])
    if invert:
        scores = -scores

    prec, rec, thresholds = precision_recall_curve(y_true, scores)

    # F1
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1)

    ax.plot(thresholds, prec[:-1], color='#1976D2', linewidth=2, label='Precision')
    ax.plot(thresholds, rec[:-1],  color='#D32F2F', linewidth=2, label='Recall')
    ax.plot(thresholds, f1[:-1],   color='#388E3C', linewidth=2, linestyle='--', label='F1')
    ax.axvline(thresholds[best_idx], color='orange', linewidth=1.5,
               linestyle=':', label=f'Best threshold={thresholds[best_idx]:.2f}  '
                                    f'F1={f1[best_idx]:.3f}')
    ax.set_xlabel('Decision Threshold (z-score)', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    n_pos = y_true.sum()
    n_tot = len(y_true)
    print(f"\n  {target} best threshold = {thresholds[best_idx]:.3f}  "
          f"Precision={prec[best_idx]:.3f}  Recall={rec[best_idx]:.3f}  "
          f"F1={f1[best_idx]:.3f}  (base rate={n_pos/n_tot*100:.1f}%)")

fig.suptitle('Step 3 — Precision / Recall / F1 vs Detection Threshold\n'
             'What threshold on val RMSE z-score gives best outlier detection?',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's3_fig6_threshold_precision_recall.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: s3_fig6_threshold_precision_recall.png")

# ── Save enriched records for downstream steps ────────────────────────────────
csv_path = os.path.join(OUT_DIR, 'records_with_dynamics.csv')
fieldnames = list(records[0].keys())
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
print(f"\n  Saved enriched records: {csv_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  STEP 3 COMPLETE -- KEY TAKEAWAYS")
print("=" * 60)

# AUC for the two detectors
for target, feature_key, invert in [
        ('Type1', 'val_rmse_zscore', False),
        ('Type2', 'val_rmse_zscore', True)]:
    y_true = np.array([1 if r['otype'] == target else 0 for r in records])
    scores = np.array([r[feature_key] for r in records])
    if invert:
        scores = -scores
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    print(f"\n  {target}  (val RMSE z-score{'  inverted' if invert else ''})")
    print(f"    ROC-AUC = {roc_auc:.4f}")
    if roc_auc > 0.85:
        print("    -> STRONG: training history alone is a reliable detector")
    elif roc_auc > 0.70:
        print("    -> MODERATE: useful signal, combine with other features")
    else:
        print("    -> WEAK: training history insufficient alone, need runtime features")

print()
print("  -> Next: Step 4 -- Trajectory feature extraction")
print(f"  -> Output files in: {OUT_DIR}/")
