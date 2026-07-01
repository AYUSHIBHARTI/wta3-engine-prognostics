# =============================================================================
# step4_trajectory_features.py
# Extract 8 trajectory features from each model's test prediction sequence.
# These are computable at inference time — no ensemble needed.
# Key question: does the SHAPE of a prediction trajectory reveal its reliability?
# =============================================================================

import json
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc

plt.style.use('default')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

RESULTS_DIR = 'results'
STEP3_DIR   = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step3')
OUT_DIR     = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step4')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES      = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
COLORS           = {'Type1': '#D32F2F', 'Type2': '#1976D2', 'Inlier': '#388E3C'}
PRED_START_CYCLE = 61
TOTAL_CYCLES     = 213
N_PRED           = TOTAL_CYCLES - PRED_START_CYCLE + 1   # 153
WINDOW           = 10

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
with open(os.path.join(RESULTS_DIR, 'all_predictions.json')) as f:
    predictions = json.load(f)

records = []
with open(os.path.join(STEP3_DIR, 'records_with_dynamics.csv')) as f:
    for row in csv.DictReader(f):
        for col in ['val_rmse','train_rmse','gap_ratio','val_rmse_zscore',
                    'gap_zscore','test_mae']:
            row[col] = float(row[col]) if row[col] != '' else np.nan
        row['outlier'] = row['outlier'] == 'True'
        records.append(row)

# Index records by (experiment, model)
rec_map = {(r['experiment'], r['model']): r for r in records}

# True RUL
y_true = np.array(predictions[0]['y_test'])
cycles = np.arange(PRED_START_CYCLE, TOTAL_CYCLES + 1)
print(f"  {len(predictions)} experiments, {N_PRED} prediction cycles each\n")

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — 8 features per (experiment, model) pair
# ═════════════════════════════════════════════════════════════════════════════

def extract_trajectory_features(preds):
    """
    preds: 1D array of length N_PRED (153 values)
    Returns dict of scalar features.
    """
    p    = np.array(preds, dtype=float)
    diff = np.diff(p)                          # p[t] - p[t-1]  length 152

    # 1. Monotonicity violations
    #    RUL should decrease — any increase is a violation
    mono_violations = int((diff > 0).sum())
    mono_rate       = mono_violations / len(diff)

    # 2. Max single-cycle jump (absolute)
    max_jump = float(np.max(np.abs(diff)))

    # 3. Mean absolute cycle-to-cycle change
    mean_abs_diff = float(np.mean(np.abs(diff)))

    # 4. Rolling variance (mean of window-variances)
    roll_vars = []
    for c in range(len(p)):
        lo = max(0, c - WINDOW + 1)
        roll_vars.append(np.var(p[lo:c+1]))
    mean_rolling_var = float(np.mean(roll_vars))

    # 5. Deviation from own moving average
    #    How much does each prediction deviate from its local trend?
    ma_devs = []
    for c in range(len(p)):
        lo  = max(0, c - WINDOW + 1)
        ma  = np.mean(p[lo:c+1])
        ma_devs.append(abs(p[c] - ma))
    mean_ma_dev = float(np.mean(ma_devs))

    # 6. Prediction slope (linear regression slope, should be ~ -1 for ideal RUL)
    t         = np.arange(len(p))
    coef      = np.polyfit(t, p, 1)
    slope     = float(coef[0])             # negative = decreasing (good)
    slope_dev = float(abs(slope - (-1.0))) # deviation from ideal slope of -1

    # 7. Slope consistency (std of local slopes across 10-cycle windows)
    local_slopes = []
    for start in range(0, len(p) - WINDOW, WINDOW // 2):
        seg  = p[start:start + WINDOW]
        ts   = np.arange(len(seg))
        c_   = np.polyfit(ts, seg, 1)
        local_slopes.append(c_[0])
    slope_consistency = float(np.std(local_slopes)) if local_slopes else 0.0

    # 8. Final prediction (prediction at the last cycle — should be near 0)
    final_pred = float(p[-1])

    return {
        'mono_violations'   : mono_violations,
        'mono_rate'         : mono_rate,
        'max_jump'          : max_jump,
        'mean_abs_diff'     : mean_abs_diff,
        'mean_rolling_var'  : mean_rolling_var,
        'mean_ma_dev'       : mean_ma_dev,
        'slope'             : slope,
        'slope_dev'         : slope_dev,
        'slope_consistency' : slope_consistency,
        'final_pred'        : final_pred,
    }

FEAT_NAMES = ['mono_violations','mono_rate','max_jump','mean_abs_diff',
              'mean_rolling_var','mean_ma_dev','slope','slope_dev',
              'slope_consistency','final_pred']

print("Extracting trajectory features for all 600 (exp x model) runs...")
for pred_row in predictions:
    exp_name = pred_row['experiment']
    for m in MODEL_NAMES:
        preds = pred_row['test_predictions'][m]
        feats = extract_trajectory_features(preds)
        key   = (exp_name, m)
        if key in rec_map:
            rec_map[key].update(feats)

print(f"  Done. Features extracted: {FEAT_NAMES}\n")

# ── Summary statistics per type ───────────────────────────────────────────────
print("=" * 75)
print("  TRAJECTORY FEATURE SUMMARY BY OUTLIER TYPE")
print("=" * 75)
print(f"\n{'Feature':<22} {'Inlier mean':>13} {'Type1 mean':>12} "
      f"{'Type2 mean':>12} {'Type1/Inlier':>13} {'Type2/Inlier':>13}")
print("-" * 80)
for feat in FEAT_NAMES:
    inlier_vals = [r[feat] for r in records if r['otype'] == 'Inlier']
    type1_vals  = [r[feat] for r in records if r['otype'] == 'Type1']
    type2_vals  = [r[feat] for r in records if r['otype'] == 'Type2']
    mu_in = np.mean(inlier_vals)
    mu_t1 = np.mean(type1_vals) if type1_vals else np.nan
    mu_t2 = np.mean(type2_vals) if type2_vals else np.nan
    ratio1 = mu_t1 / mu_in if mu_in != 0 and not np.isnan(mu_t1) else np.nan
    ratio2 = mu_t2 / mu_in if mu_in != 0 and not np.isnan(mu_t2) else np.nan
    print(f"  {feat:<20} {mu_in:>13.3f} {mu_t1:>12.3f} {mu_t2:>12.3f} "
          f"{ratio1:>12.2f}x {ratio2:>12.2f}x")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Feature distributions: Inlier vs Type1 vs Type2
#            One subplot per feature (2 rows x 5 cols)
# ═════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Feature distributions by type...")
display_feats = ['mono_violations','max_jump','mean_rolling_var',
                 'mean_ma_dev','slope_dev','slope_consistency',
                 'final_pred','mean_abs_diff']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feat in enumerate(display_feats):
    ax = axes[i]
    for otype, color in [('Inlier','#388E3C'),('Type1','#D32F2F'),('Type2','#1976D2')]:
        vals = [r[feat] for r in records if r['otype'] == otype]
        # clip extreme values for readability
        p95  = np.percentile(vals, 95)
        vals_c = np.clip(vals, 0, p95 * 1.5)
        ax.hist(vals_c, bins=30, alpha=0.6, color=color, density=True,
                label=f"{otype}  μ={np.mean(vals):.2f}")
    ax.set_title(feat.replace('_', ' '), fontsize=9)
    ax.set_xlabel('Value', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.2)

fig.suptitle('Step 4 — Trajectory Feature Distributions: Inlier vs Type 1 vs Type 2\n'
             '(all models combined — features computable from a single model\'s predictions)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's4_fig1_feature_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s4_fig1_feature_distributions.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ROC curves: each trajectory feature vs Type1 and Type2
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 2: ROC curves per feature...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
line_styles = ['-','--','-.',':', '-','--','-.',':', '-','--']

for ax, target, title in [
    (axes[0], 'Type1', 'Type 1 (Bad — XGB failures)'),
    (axes[1], 'Type2', 'Type 2 (Divergent — Transformer)')]:

    y_true_bin = np.array([1 if r['otype'] == target else 0 for r in records])
    results_auc = []

    for feat in FEAT_NAMES:
        scores = np.array([r[feat] for r in records])
        fpr, tpr, _ = roc_curve(y_true_bin, scores)
        roc_auc     = auc(fpr, tpr)
        if roc_auc < 0.5:
            fpr, tpr, _ = roc_curve(y_true_bin, -scores)
            roc_auc     = auc(fpr, tpr)
        results_auc.append((feat, fpr, tpr, roc_auc))

    results_auc.sort(key=lambda x: x[3], reverse=True)

    for (feat, fpr, tpr, roc_auc), ls in zip(results_auc, line_styles):
        ax.plot(fpr, tpr, ls, linewidth=1.8,
                label=f'{feat.replace("_"," "):<22}  AUC={roc_auc:.3f}')

    ax.plot([0,1],[0,1], 'k--', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.2)

    print(f"\n  {target} — AUC ranking:")
    for feat, _, _, roc_auc in results_auc:
        bar = '#' * int(roc_auc * 20)
        print(f"    {feat:<22} AUC={roc_auc:.4f}  {bar}")

fig.suptitle('Step 4 — ROC Curves: Trajectory Features as Single-Model Detectors\n'
             'Higher AUC = feature alone can distinguish outliers from inliers',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's4_fig2_roc_per_feature.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: s4_fig2_roc_per_feature.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Feature correlation matrix (all runs)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 3: Feature correlation matrix...")
feat_matrix = np.array([[r[f] for f in FEAT_NAMES] for r in records])
corr        = np.corrcoef(feat_matrix.T)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson Correlation')
ax.set_xticks(range(len(FEAT_NAMES)))
ax.set_yticks(range(len(FEAT_NAMES)))
labels = [f.replace('_','\n') for f in FEAT_NAMES]
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for i in range(len(FEAT_NAMES)):
    for j in range(len(FEAT_NAMES)):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                fontsize=7, color='white' if abs(corr[i,j]) > 0.6 else 'black')
ax.set_title('Step 4 — Trajectory Feature Correlation Matrix\n'
             '(high correlation = redundant features — keep only one per group)',
             fontsize=11, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's4_fig3_feature_correlation.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s4_fig3_feature_correlation.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Top 2 features scatter: Type1 and Type2 separation
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 4: 2D scatter of top features...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Type1: top 2 features from ROC analysis
# Type2: top 2 features from ROC analysis
for ax, target, fx, fy, title in [
    (axes[0], 'Type1', 'mean_rolling_var', 'val_rmse_zscore',
     'Type 1 Detection Space'),
    (axes[1], 'Type2', 'slope_dev', 'val_rmse_zscore',
     'Type 2 Detection Space')]:

    for otype in ['Inlier', 'Type1', 'Type2']:
        recs = [r for r in records if r['otype'] == otype]
        xs   = np.clip([r[fx] for r in recs], 0, np.percentile([r[fx] for r in records], 98))
        ys   = [r[fy] for r in recs]
        ax.scatter(xs, ys, c=COLORS[otype], s=30, alpha=0.75,
                   label=f'{otype} (n={len(recs)})', zorder=2+list(COLORS).index(otype))

    ax.set_xlabel(fx.replace('_', ' '), fontsize=10)
    ax.set_ylabel(fy.replace('_', ' '), fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

fig.suptitle('Step 4 — 2D Feature Space: Can We Draw a Separation Boundary?\n'
             '(if types cluster separately, a simple classifier will work well)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's4_fig4_2d_feature_scatter.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s4_fig4_2d_feature_scatter.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Per-model trajectory: mean prediction curve for each type
#            Shows where in the engine lifetime each type diverges
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5: Mean prediction trajectories per model...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax = axes[j]

    for otype, color, lw in [('Inlier','#388E3C',1.5),
                               ('Type1','#D32F2F',2.0),
                               ('Type2','#1976D2',2.0)]:
        exp_names = [r['experiment'] for r in records
                     if r['model'] == m and r['otype'] == otype]
        if not exp_names:
            continue

        # Collect prediction arrays for these experiments
        pred_arrays = []
        for pred_row in predictions:
            if pred_row['experiment'] in exp_names:
                pred_arrays.append(pred_row['test_predictions'][m])

        if pred_arrays:
            arr       = np.array(pred_arrays)
            mean_pred = arr.mean(axis=0)
            std_pred  = arr.std(axis=0)
            ax.plot(cycles, mean_pred, color=color, linewidth=lw,
                    label=f'{otype} (n={len(pred_arrays)})', alpha=0.9)
            ax.fill_between(cycles, mean_pred - std_pred, mean_pred + std_pred,
                            color=color, alpha=0.08)

    # True RUL
    ax.plot(cycles, y_true, 'k--', linewidth=1, alpha=0.5, label='True RUL')
    ax.set_title(m, fontsize=11)
    ax.set_xlabel('Cycle', fontsize=9)
    ax.set_ylabel('Predicted RUL', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

fig.suptitle('Step 4 — Mean Prediction Trajectories by Outlier Type per Model\n'
             '(shaded = ±1 std — where do outlier predictions visibly diverge from true RUL?)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's4_fig5_mean_trajectories.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s4_fig5_mean_trajectories.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Feature values per model, coloured by type (boxplot grid)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 6: Feature boxplots per model...")
top_feats = ['mean_rolling_var', 'slope_dev', 'mono_violations', 'max_jump']

fig, axes = plt.subplots(len(top_feats), len(MODEL_NAMES),
                          figsize=(16, 12))

for fi, feat in enumerate(top_feats):
    for mi, m in enumerate(MODEL_NAMES):
        ax = axes[fi, mi]
        data_b, tick_b, col_b = [], [], []
        for otype, color in [('Inlier','#388E3C'),
                              ('Type1','#D32F2F'),
                              ('Type2','#1976D2')]:
            recs = [r for r in records
                    if r['model'] == m and r['otype'] == otype]
            if recs:
                vals = np.clip([r[feat] for r in recs],
                               0, np.percentile([r[feat] for r in records
                                                 if r['model']==m], 99))
                data_b.append(vals)
                tick_b.append(otype[:3])
                col_b.append(color)

        bp = ax.boxplot(data_b, tick_labels=tick_b,
                        patch_artist=True, notch=False)
        for patch, color in zip(bp['boxes'], col_b):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for med in bp['medians']:
            med.set_color('black')
            med.set_linewidth(1.5)

        if mi == 0:
            ax.set_ylabel(feat.replace('_','\n'), fontsize=7)
        if fi == 0:
            ax.set_title(m, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(axis='y', alpha=0.2)

fig.suptitle('Step 4 — Top Trajectory Features by Model and Outlier Type\n'
             '(rows = features, columns = models)',
             fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUT_DIR, 's4_fig6_feature_boxplots_per_model.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s4_fig6_feature_boxplots_per_model.png")

# ── Save enriched records ─────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, 'records_with_trajectory.csv')
all_keys = list(records[0].keys())
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore', restval='')
    writer.writeheader()
    writer.writerows(records)
print(f"\n  Saved enriched records: {csv_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  STEP 4 COMPLETE -- KEY TAKEAWAYS")
print("=" * 65)
print()
print("  Best trajectory features for Type 1 detection (AUC ranking):")

y_t1 = np.array([1 if r['otype']=='Type1' else 0 for r in records])
y_t2 = np.array([1 if r['otype']=='Type2' else 0 for r in records])

def best_auc(scores, y):
    fpr, tpr, _ = roc_curve(y, scores)
    a = auc(fpr, tpr)
    if a < 0.5:
        fpr, tpr, _ = roc_curve(y, -scores)
        a = auc(fpr, tpr)
    return a

auc_results = []
for feat in FEAT_NAMES:
    scores = np.array([r[feat] for r in records])
    auc_results.append((feat, best_auc(scores, y_t1), best_auc(scores, y_t2)))

auc_results.sort(key=lambda x: x[1], reverse=True)
print(f"  {'Feature':<24} {'AUC vs Type1':>13} {'AUC vs Type2':>13}")
print("  " + "-" * 52)
for feat, a1, a2 in auc_results:
    flag1 = " STRONG"   if a1 > 0.80 else (" mod" if a1 > 0.65 else "")
    flag2 = " STRONG"   if a2 > 0.80 else (" mod" if a2 > 0.65 else "")
    print(f"  {feat:<24} {a1:>8.4f}{flag1:<8}  {a2:>8.4f}{flag2}")

print()
print("  -> Next: Step 5 -- Error magnitude by RUL zone per outlier type")
print(f"  -> Output files in: {OUT_DIR}/")
