"""
Step 9: Inter-Model Error Correlation Matrix
============================================
Goal: Examine how prediction errors correlate ACROSS models within the same
experiment. If a single model is an outlier, its errors should be poorly
correlated (or anti-correlated) with the other 5 models in its experiment.

Key ideas:
  - For each experiment, compute per-cycle error for all 6 models
  - Build 6x6 Pearson correlation matrix of errors
  - Summarise each (experiment, model) run by:
      mean_corr_with_others  -- average corr with the other 5 models
      min_corr_with_others   -- worst (most discordant) corr with any model
      rank_in_experiment     -- 1 = most correlated (least divergent), 6 = most divergent
  - Compare those summary stats across Inlier / Type1 / Type2
  - ROC analysis: can mean_corr_with_others detect outliers without ensemble labels?
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc as sk_auc
from scipy import stats

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.join(BASE, 'results', 'outlier_analysis', 'step9')
os.makedirs(RES, exist_ok=True)

STEP1_CSV = os.path.join(BASE, 'results', 'outlier_analysis', 'step1',
                          'outlier_type_labels.csv')
CHK1  = os.path.join(BASE, 'results', 'step1_checkpoint.json')
CHK2  = os.path.join(BASE, 'results', 'step2_checkpoint.json')
PREDS = os.path.join(BASE, 'results', 'all_predictions.json')

MODELS = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
TYPE_COLORS = {'Inlier': '#2196F3', 'Type1': '#c62828', 'Type2': '#e65100'}

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data...")
labels_df = pd.read_csv(STEP1_CSV)

with open(CHK1) as f: ck1 = json.load(f)
with open(CHK2) as f: ck2 = json.load(f)
with open(PREDS, 'r') as f:
    all_preds_raw = json.load(f)
# Convert list of experiment dicts to {(exp_idx, model): preds}
all_preds = {}
for item in all_preds_raw:
    ei = item['experiment']
    for mname, preds in item['test_predictions'].items():
        all_preds[(ei, mname)] = preds

test_engine   = ck1['test_engine']
total_cycles  = ck1['total_cycles']
calibration   = ck1['calibration']
pred_points   = ck1['prediction_points']

true_rul = np.array([max(0, total_cycles - calibration - i)
                     for i in range(pred_points)], dtype=float)

# experiment IDs are strings like 'Exp_001'
exp_ids = sorted(set(k for k, _ in all_preds.keys()))
n_exp   = len(exp_ids)
exp_to_idx = {eid: i for i, eid in enumerate(exp_ids)}

# ── build error array: [n_exp, n_models, n_cycles] ────────────────────────
print("Building error arrays...")
err_arr = np.full((n_exp, len(MODELS), pred_points), np.nan)

for (eid, mname), preds in all_preds.items():
    ei = exp_to_idx[eid]
    mi = MODELS.index(mname) if mname in MODELS else -1
    if mi < 0:
        continue
    p = np.array(preds, dtype=float)
    if len(p) == pred_points:
        err_arr[ei, mi, :] = np.abs(p - true_rul)

# ── compute per-run summary stats ─────────────────────────────────────────
print("Computing inter-model correlation statistics...")

records = []

for eid in exp_ids:
    ei  = exp_to_idx[eid]
    mat = err_arr[ei]  # (6, 153)

    valid = [mi for mi in range(len(MODELS)) if not np.all(np.isnan(mat[mi]))]
    if len(valid) < 2:
        continue

    # 6x6 Pearson correlation matrix
    corr_matrix = np.full((len(MODELS), len(MODELS)), np.nan)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i in valid and j in valid:
                r, _ = stats.pearsonr(mat[i], mat[j])
                corr_matrix[i, j] = r

    # per-model summary
    for mi, mname in enumerate(MODELS):
        if mi not in valid:
            continue

        other_corrs = [corr_matrix[mi, j] for j in range(len(MODELS))
                       if j != mi and not np.isnan(corr_matrix[mi, j])]
        if not other_corrs:
            continue

        mean_corr = float(np.nanmean(other_corrs))
        min_corr  = float(np.nanmin(other_corrs))
        max_corr  = float(np.nanmax(other_corrs))
        std_corr  = float(np.nanstd(other_corrs))

        # rank: 1 = most correlated (least divergent), 6 = most divergent
        all_means = []
        for ki in range(len(MODELS)):
            if ki not in valid:
                all_means.append((-np.inf, ki))
                continue
            oc = [corr_matrix[ki, j] for j in range(len(MODELS))
                  if j != ki and not np.isnan(corr_matrix[ki, j])]
            all_means.append((float(np.nanmean(oc)) if oc else -np.inf, ki))
        all_means_sorted = sorted(all_means, key=lambda x: -x[0])
        rank = [r for r, (_, ki) in enumerate(all_means_sorted, 1) if ki == mi][0]

        row   = labels_df[(labels_df['experiment'] == eid) &
                          (labels_df['model'] == mname)]
        otype = row['otype'].values[0] if len(row) else 'Inlier'

        records.append(dict(
            experiment=eid, model=mname, outlier_type=otype,
            mean_corr_with_others=mean_corr,
            min_corr_with_others=min_corr,
            max_corr_with_others=max_corr,
            std_corr_with_others=std_corr,
            rank_in_experiment=rank,
        ))

df = pd.DataFrame(records)
print(f"  Records: {len(df)}")

# ── summary statistics ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  INTER-MODEL CORRELATION SUMMARY BY OUTLIER TYPE")
print("="*70)
print(f"\n{'Type':<10} {'N':>5} {'mean_corr':>12} {'min_corr':>12} {'rank':>8}")
print("-"*50)
for otype in ['Inlier', 'Type1', 'Type2']:
    sub = df[df['outlier_type'] == otype]
    if len(sub) == 0:
        continue
    print(f"{otype:<10} {len(sub):>5} "
          f"{sub['mean_corr_with_others'].mean():>12.4f} "
          f"{sub['min_corr_with_others'].mean():>12.4f} "
          f"{sub['rank_in_experiment'].mean():>8.2f}")

# ── ROC analysis ───────────────────────────────────────────────────────────
print("\n  ROC-AUC for inter-model correlation features:")

def compute_roc_auc(scores, y_true):
    """scores: higher = more suspicious (outlier)"""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return sk_auc(fpr, tpr), fpr, tpr

features_for_roc = {
    'neg_mean_corr': -df['mean_corr_with_others'],
    'neg_min_corr':  -df['min_corr_with_others'],
    'rank':           df['rank_in_experiment'],
    'std_corr':       df['std_corr_with_others'],
}

roc_results = {}
for fname, scores in features_for_roc.items():
    # Type1 detection
    y1 = (df['outlier_type'] == 'Type1').astype(int)
    if y1.sum() > 0:
        a1, fpr1, tpr1 = compute_roc_auc(scores, y1)
    else:
        a1, fpr1, tpr1 = 0.5, [0,1], [0,1]

    # Type2 detection
    y2 = (df['outlier_type'] == 'Type2').astype(int)
    if y2.sum() > 0:
        a2, fpr2, tpr2 = compute_roc_auc(scores, y2)
    else:
        a2, fpr2, tpr2 = 0.5, [0,1], [0,1]

    roc_results[fname] = dict(auc1=a1, fpr1=fpr1, tpr1=tpr1,
                               auc2=a2, fpr2=fpr2, tpr2=tpr2)
    print(f"  {fname:<22}  Type1 AUC={a1:.4f}  Type2 AUC={a2:.4f}")

# ── Figure 1: boxplots of mean_corr and rank by type ─────────────────────
print("\nFigure 1: Distribution of inter-model correlation by type...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Step 9: Inter-Model Error Correlation by Outlier Type",
             fontsize=14, fontweight='bold', y=0.98)

for ax, (feat, label) in zip(axes, [
        ('mean_corr_with_others', 'Mean Corr with Other 5 Models'),
        ('min_corr_with_others',  'Min Corr with Any Single Model'),
        ('rank_in_experiment',    'Rank in Experiment\n(6=most divergent)'),
]):
    data_by_type = [df[df['outlier_type'] == t][feat].dropna().values
                    for t in ['Inlier', 'Type1', 'Type2']]
    bp = ax.boxplot(data_by_type, tick_labels=['Inlier', 'Type1', 'Type2'],
                    patch_artist=True, notch=False, showfliers=True)
    for patch, color in zip(bp['boxes'], [TYPE_COLORS[t] for t in ['Inlier','Type1','Type2']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(label, fontsize=11)
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f8f8f8')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's9_fig1_corr_distributions.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s9_fig1_corr_distributions.png")

# ── Figure 2: average 6x6 correlation matrix per type ────────────────────
print("Figure 2: Average correlation matrix per type...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Step 9: Average Inter-Model Error Correlation Matrix by Type",
             fontsize=14, fontweight='bold', y=0.98)

for ax, otype in zip(axes, ['Inlier', 'Type1', 'Type2']):
    sub_exp = df[df['outlier_type'] == otype]['experiment'].unique()
    corr_sum = np.zeros((len(MODELS), len(MODELS)))
    count = 0
    for eid in sub_exp:
        ei  = exp_to_idx[eid]
        mat = err_arr[ei]
        valid = [mi for mi in range(len(MODELS)) if not np.all(np.isnan(mat[mi]))]
        if len(valid) < 2:
            continue
        cm = np.full((len(MODELS), len(MODELS)), np.nan)
        for i in range(len(MODELS)):
            for j in range(len(MODELS)):
                if i == j:
                    cm[i, j] = 1.0
                elif i in valid and j in valid:
                    r, _ = stats.pearsonr(mat[i], mat[j])
                    cm[i, j] = r
        corr_sum += np.nan_to_num(cm, nan=0.0)
        count += 1
    avg_corr = corr_sum / max(count, 1)

    im = ax.imshow(avg_corr, vmin=-1, vmax=1, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(MODELS)))
    ax.set_yticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(MODELS, fontsize=9)
    ax.set_title(f'{otype} (n_exp={len(sub_exp)})', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            v = avg_corr[i, j]
            txt_color = 'black' if abs(v) < 0.5 else 'white'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color=txt_color)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's9_fig2_avg_corr_matrix.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s9_fig2_avg_corr_matrix.png")

# ── Figure 3: ROC curves ──────────────────────────────────────────────────
print("Figure 3: ROC curves...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 9: ROC Curves — Inter-Model Correlation Features",
             fontsize=14, fontweight='bold', y=0.98)

feat_styles = {
    'neg_mean_corr': ('-',  '#c62828', 'Neg Mean Corr'),
    'neg_min_corr':  ('--', '#e65100', 'Neg Min Corr'),
    'rank':          ('-.',  '#6a1b9a', 'Rank (6=worst)'),
    'std_corr':      (':',  '#1565c0', 'Std Corr'),
}

for ax, (type_key, title) in zip(axes, [('1', 'Type1 Detection'), ('2', 'Type2 Detection')]):
    ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random (0.50)')
    for fname, (ls, color, label) in feat_styles.items():
        res = roc_results[fname]
        fpr = res[f'fpr{type_key}']
        tpr = res[f'tpr{type_key}']
        a   = res[f'auc{type_key}']
        ax.plot(fpr, tpr, linestyle=ls, color=color, lw=2,
                label=f'{label} (AUC={a:.3f})')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3); ax.set_facecolor('#f8f8f8')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's9_fig3_roc_curves.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s9_fig3_roc_curves.png")

# ── Figure 4: scatter — mean_corr vs mean_abs_error coloured by type ──────
print("Figure 4: Scatter — mean_corr vs mean absolute error...")
# load trajectory records for mean_abs_err
step4_csv = os.path.join(BASE, 'results', 'outlier_analysis', 'step4',
                          'records_with_trajectory.csv')
traj_df = pd.read_csv(step4_csv)

merged = df.merge(traj_df[['experiment','model','val_rmse']],
                  on=['experiment','model'], how='left')

fig, ax = plt.subplots(figsize=(10, 6))
for otype in ['Inlier', 'Type2', 'Type1']:
    sub = merged[merged['outlier_type'] == otype]
    ax.scatter(sub['mean_corr_with_others'], sub['val_rmse'],
               c=TYPE_COLORS[otype], label=otype,
               alpha=0.55, s=40 if otype == 'Inlier' else 80,
               edgecolors='none')

ax.set_xlabel('Mean Correlation with Other 5 Models', fontsize=12)
ax.set_ylabel('Val RMSE', fontsize=12)
ax.set_title('Step 9: Val RMSE vs Inter-Model Error Correlation\n'
             '(Outliers should cluster low-correlation OR high-RMSE)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_facecolor('#f8f8f8')
plt.tight_layout()
plt.savefig(os.path.join(RES, 's9_fig4_scatter_corr_vs_rmse.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s9_fig4_scatter_corr_vs_rmse.png")

# ── Figure 5: per-model mean_corr by type ─────────────────────────────────
print("Figure 5: Per-model mean_corr distributions...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Step 9: Mean Correlation with Other Models — Per Model Type",
             fontsize=14, fontweight='bold', y=0.98)

for ax, mname in zip(axes.flat, MODELS):
    sub = df[df['model'] == mname]
    data_by_type = [sub[sub['outlier_type'] == t]['mean_corr_with_others'].dropna().values
                    for t in ['Inlier', 'Type1', 'Type2']]
    labels = ['Inlier', 'Type1', 'Type2']
    colors = [TYPE_COLORS[t] for t in labels]
    non_empty = [(d, l, c) for d, l, c in zip(data_by_type, labels, colors) if len(d) > 0]
    if not non_empty:
        ax.set_visible(False)
        continue
    bp = ax.boxplot([x[0] for x in non_empty],
                    tick_labels=[x[1] for x in non_empty],
                    patch_artist=True, notch=False)
    for patch, (_, _, color) in zip(bp['boxes'], non_empty):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(mname, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Corr with Others')
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f8f8f8')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's9_fig5_per_model_corr.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s9_fig5_per_model_corr.png")

# ── Figure 6: rank distribution heatmap (model x rank) ───────────────────
print("Figure 6: Rank distribution heatmap...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Step 9: Rank Distribution by Model (1=most correlated, 6=most divergent)",
             fontsize=13, fontweight='bold', y=0.98)

for ax, otype in zip(axes, ['Inlier', 'Type1', 'Type2']):
    sub = df[df['outlier_type'] == otype]
    heat = np.zeros((len(MODELS), 6))
    for mi, mname in enumerate(MODELS):
        msub = sub[sub['model'] == mname]
        for rank in range(1, 7):
            heat[mi, rank-1] = (msub['rank_in_experiment'] == rank).sum()
    # normalise row-wise
    row_sums = heat.sum(axis=1, keepdims=True)
    heat_norm = heat / np.where(row_sums == 0, 1, row_sums)

    im = ax.imshow(heat_norm, vmin=0, vmax=0.5, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'Rank {r}' for r in range(1, 7)], rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10)
    ax.set_title(f'{otype}', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, label='Proportion')
    for i in range(len(MODELS)):
        for j in range(6):
            v = heat_norm[i, j]
            txt_color = 'black' if v < 0.3 else 'white'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color=txt_color)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's9_fig6_rank_heatmap.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s9_fig6_rank_heatmap.png")

# ── Save CSV ───────────────────────────────────────────────────────────────
out_csv = os.path.join(RES, 'records_step9.csv')
df.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv}")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 9 COMPLETE -- KEY TAKEAWAYS")
print("="*65)

for otype in ['Inlier', 'Type1', 'Type2']:
    sub = df[df['outlier_type'] == otype]
    if len(sub) == 0:
        continue
    mc = sub['mean_corr_with_others'].mean()
    mnc = sub['min_corr_with_others'].mean()
    rk = sub['rank_in_experiment'].mean()
    print(f"\n  {otype} ({len(sub)} runs):")
    print(f"    mean_corr_with_others = {mc:.4f}")
    print(f"    min_corr_with_others  = {mnc:.4f}")
    print(f"    avg rank              = {rk:.2f}")

best_f1 = max(roc_results, key=lambda x: roc_results[x]['auc1'])
best_f2 = max(roc_results, key=lambda x: roc_results[x]['auc2'])
print(f"\n  Best detector for Type1: {best_f1} (AUC={roc_results[best_f1]['auc1']:.4f})")
print(f"  Best detector for Type2: {best_f2} (AUC={roc_results[best_f2]['auc2']:.4f})")
print(f"\n  -> Next: Step 10 -- Consensus analysis (divergence from group)")
print(f"  -> Output files in: {RES}/")
