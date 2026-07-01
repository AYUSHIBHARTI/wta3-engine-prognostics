"""
Step 12: Simple Rule-Based Single-Model Outlier Indicator
==========================================================
Goal: Design the simplest possible rule-based detector that flags
outlier models without needing the ensemble. This should be:
  (a) Interpretable — a human can understand why a flag was raised
  (b) Threshold-based — no ML model required
  (c) Split by type — different rules for Type1 vs Type2

Design strategy:
  - Type1 (XGB underfitting): trajectory diverges from consensus,
    final_pred high, slope deviates from -1 (expected degradation rate)
    Rule: mean_abs_div > T1 OR slope_dev > T2
  - Type2 (Transformer over-performing): too few monotonicity violations
    (too smooth / over-regularised), val_rmse very low
    Rule: mono_rate < T3 AND val_rmse < T4

Threshold selection: F1-optimal from training data (all 600 runs).
Because we're doing a retrospective analysis (not building a deployed
system), we can choose thresholds on the full dataset and report
precision, recall, F1.

Also compute:
  - Combined single-score: weighted combination of top features
  - Performance on Type1, Type2, and "any outlier" tasks
  - Compare to using val_rmse alone (naive baseline)
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc as sk_auc, precision_recall_curve,
                              f1_score, precision_score, recall_score,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(BASE, 'results', 'outlier_analysis', 'step12')
os.makedirs(RES, exist_ok=True)

STEP11_CSV  = os.path.join(BASE, 'results', 'outlier_analysis', 'step11',
                            'records_step11.csv')
SEL_FEATS   = os.path.join(BASE, 'results', 'outlier_analysis', 'step11',
                            'selected_features.json')

TYPE_COLORS = {'Inlier': '#2196F3', 'Type1': '#c62828', 'Type2': '#e65100'}

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(STEP11_CSV)
with open(SEL_FEATS) as f:
    sel = json.load(f)

# normalise type column
for col in ['outlier_type', 'otype']:
    if col in df.columns and col != 'outlier_type':
        df.rename(columns={col: 'outlier_type'}, inplace=True)

print(f"  Records: {len(df)}, types: {df['outlier_type'].value_counts().to_dict()}")

# ── Helper: find F1-optimal threshold ─────────────────────────────────────
def find_best_threshold(scores, y_true, metric='f1'):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precision * recall / np.where((precision + recall) == 0, 1,
                                             precision + recall)
    best_idx = np.argmax(f1s[:-1])  # last element has no threshold
    best_t   = float(thresholds[best_idx])
    best_f1  = float(f1s[best_idx])
    return best_t, best_f1, precision[best_idx], recall[best_idx]

def evaluate_rule(pred_labels, true_labels, label='Rule'):
    tp = int(((pred_labels == 1) & (true_labels == 1)).sum())
    fp = int(((pred_labels == 1) & (true_labels == 0)).sum())
    fn = int(((pred_labels == 0) & (true_labels == 1)).sum())
    tn = int(((pred_labels == 0) & (true_labels == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"  {label:<40}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
          f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return dict(label=label, precision=prec, recall=rec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)

# ── Impute missing values ──────────────────────────────────────────────────
feat_cols = sel['all_features']
for col in feat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ── RULE SET 1: Type1 detection ───────────────────────────────────────────
print("\n" + "="*70)
print("  RULE-BASED DETECTION: TYPE1 (XGB Underfitting)")
print("="*70)

y1 = (df['outlier_type'] == 'Type1').astype(int)

results = []

# R1a: mean_abs_div threshold
score_div = df['mean_abs_div']
t1a, f1a, p1a, r1a = find_best_threshold(score_div, y1)
pred_r1a = (score_div >= t1a).astype(int)
results.append(evaluate_rule(pred_r1a, y1, f'R1a: mean_abs_div >= {t1a:.2f}'))

# R1b: slope_dev threshold
score_sdev = df['slope_dev']
t1b, f1b, p1b, r1b = find_best_threshold(score_sdev, y1)
pred_r1b = (score_sdev >= t1b).astype(int)
results.append(evaluate_rule(pred_r1b, y1, f'R1b: slope_dev >= {t1b:.4f}'))

# R1c: val_rmse_zscore threshold (training signal)
score_zscore = df['val_rmse_zscore']
t1c, f1c, p1c, r1c = find_best_threshold(score_zscore, y1)
pred_r1c = (score_zscore >= t1c).astype(int)
results.append(evaluate_rule(pred_r1c, y1, f'R1c: val_rmse_zscore >= {t1c:.4f}'))

# R1d: combined AND rule
pred_r1d = ((score_div >= t1a) | (score_sdev >= t1b)).astype(int)
results.append(evaluate_rule(pred_r1d, y1, 'R1d: mean_abs_div OR slope_dev'))

# R1e: baseline — val_rmse threshold
score_rmse = df['val_rmse']
t1e, f1e, _, _ = find_best_threshold(score_rmse, y1)
# val_rmse is LOW for Type1? Let's check
print(f"\n  Note: Type1 mean val_rmse = {df[y1==1]['val_rmse'].mean():.2f}, "
      f"Inlier = {df[y1==0]['val_rmse'].mean():.2f}")
pred_r1e = (score_rmse >= t1e).astype(int)
results.append(evaluate_rule(pred_r1e, y1, f'Baseline: val_rmse >= {t1e:.2f}'))

# ── RULE SET 2: Type2 detection ───────────────────────────────────────────
print("\n" + "="*70)
print("  RULE-BASED DETECTION: TYPE2 (Transformer Over-Performing)")
print("="*70)

y2 = (df['outlier_type'] == 'Type2').astype(int)

# R2a: mono_rate — too LOW = suspicious
score_mono_inv = -df['mono_rate']  # invert: low mono = high score
t2a, f2a, _, _ = find_best_threshold(score_mono_inv, y2)
pred_r2a = (score_mono_inv >= t2a).astype(int)
results.append(evaluate_rule(pred_r2a, y2,
               f'R2a: mono_rate <= {-t2a:.4f}'))

# R2b: val_rmse low (requires training info)
score_rmse_inv = -df['val_rmse']
t2b, f2b, _, _ = find_best_threshold(score_rmse_inv, y2)
pred_r2b = (score_rmse_inv >= t2b).astype(int)
results.append(evaluate_rule(pred_r2b, y2,
               f'R2b: val_rmse <= {-t2b:.2f}'))

# R2c: max_jump low (too smooth)
score_jump_inv = -df['max_jump']
t2c, f2c, _, _ = find_best_threshold(score_jump_inv, y2)
pred_r2c = (score_jump_inv >= t2c).astype(int)
results.append(evaluate_rule(pred_r2c, y2,
               f'R2c: max_jump <= {-t2c:.2f}'))

# R2d: combined — mono AND val_rmse
pred_r2d = (pred_r2a & pred_r2b).astype(int)
results.append(evaluate_rule(pred_r2d, y2,
               'R2d: low mono_rate AND low val_rmse'))

# R2e: combined OR
pred_r2e = (pred_r2a | pred_r2b).astype(int)
results.append(evaluate_rule(pred_r2e, y2,
               'R2e: low mono_rate OR low val_rmse'))

# ── RULE SET 3: Combined any-outlier ─────────────────────────────────────
print("\n" + "="*70)
print("  RULE-BASED DETECTION: ANY OUTLIER")
print("="*70)

y_any = ((df['outlier_type'] != 'Inlier')).astype(int)

# Combined rule: flag if either Type1 OR Type2 rule fires
pred_combined = (pred_r1d | pred_r2a).astype(int)
results.append(evaluate_rule(pred_combined, y_any,
               'Combined: Type1-rule OR Type2-rule'))

# Naive: val_rmse outside [low, high]
low_t  = df[y2==1]['val_rmse'].quantile(0.10)
high_t = df[y1==1]['val_rmse'].quantile(0.10)
pred_naive = ((df['val_rmse'] <= low_t) | (df['val_rmse'] >= high_t)).astype(int)
results.append(evaluate_rule(pred_naive, y_any,
               f'Naive: val_rmse outside [{low_t:.1f}, {high_t:.1f}]'))

# ── Figure 1: ROC comparison ──────────────────────────────────────────────
print("\nFigure 1: ROC curves for rule features...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Step 12: ROC Curves for Rule-Based Features",
             fontsize=14, fontweight='bold', y=0.98)

feat_styles_t1 = [
    (score_div,    '-',  '#c62828', f'mean_abs_div (thresh={t1a:.1f})'),
    (score_sdev,   '--', '#e65100', f'slope_dev (thresh={t1b:.3f})'),
    (score_zscore, '-.', '#6a1b9a', f'val_rmse_zscore (thresh={t1c:.2f})'),
    (score_rmse,   ':',  '#1565c0', f'val_rmse (thresh={t1e:.1f})'),
]
feat_styles_t2 = [
    (score_mono_inv,  '-',  '#c62828', f'neg mono_rate (thresh={t2a:.3f})'),
    (score_rmse_inv,  '--', '#e65100', f'neg val_rmse (thresh={t2b:.2f})'),
    (score_jump_inv,  '-.', '#6a1b9a', f'neg max_jump (thresh={t2c:.2f})'),
]

for ax, y_true, feat_styles, title in zip(axes,
        [y1, y2],
        [feat_styles_t1, feat_styles_t2],
        ['Type1 Detection', 'Type2 Detection']):
    ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random (0.50)')
    for scores, ls, color, label in feat_styles:
        fpr, tpr, _ = roc_curve(y_true, scores)
        a = sk_auc(fpr, tpr)
        if a < 0.5:
            a = 1.0 - a
        ax.plot(fpr, tpr, linestyle=ls, color=color, lw=2,
                label=f'{label}\n(AUC={a:.3f})')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3); ax.set_facecolor('#f8f8f8')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's12_fig1_roc_curves.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s12_fig1_roc_curves.png")

# ── Figure 2: Decision boundary visualisation ─────────────────────────────
print("Figure 2: Decision boundaries...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Step 12: Rule-Based Decision Boundaries",
             fontsize=14, fontweight='bold', y=0.98)

# Type1: mean_abs_div vs slope_dev
ax = axes[0]
for otype in ['Inlier', 'Type2', 'Type1']:
    sub = df[df['outlier_type'] == otype]
    ax.scatter(sub['mean_abs_div'], sub['slope_dev'],
               c=TYPE_COLORS[otype], label=otype,
               alpha=0.5, s=40 if otype == 'Inlier' else 80,
               edgecolors='none')
ax.axvline(t1a, color='#c62828', lw=2, linestyle='--',
           label=f'mean_abs_div={t1a:.1f}')
ax.axhline(t1b, color='#e65100', lw=2, linestyle='-.',
           label=f'slope_dev={t1b:.3f}')
ax.set_xlabel('Mean |Divergence| from Consensus', fontsize=11)
ax.set_ylabel('Slope Deviation (|slope - (-1)|)', fontsize=11)
ax.set_title('Type1 Decision Region', fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3); ax.set_facecolor('#f8f8f8')

# Type2: mono_rate vs val_rmse
ax = axes[1]
for otype in ['Inlier', 'Type1', 'Type2']:
    sub = df[df['outlier_type'] == otype]
    ax.scatter(sub['mono_rate'], sub['val_rmse'],
               c=TYPE_COLORS[otype], label=otype,
               alpha=0.5, s=40 if otype == 'Inlier' else 80,
               edgecolors='none')
ax.axvline(-t2a, color='#c62828', lw=2, linestyle='--',
           label=f'mono_rate={-t2a:.3f}')
ax.axhline(-t2b, color='#e65100', lw=2, linestyle='-.',
           label=f'val_rmse={-t2b:.2f}')
ax.set_xlabel('Monotonicity Rate (violations / total)', fontsize=11)
ax.set_ylabel('Val RMSE', fontsize=11)
ax.set_title('Type2 Decision Region', fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3); ax.set_facecolor('#f8f8f8')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's12_fig2_decision_boundaries.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s12_fig2_decision_boundaries.png")

# ── Figure 3: Precision–Recall curves ────────────────────────────────────
print("Figure 3: Precision-Recall curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Step 12: Precision-Recall Curves for Rule Features",
             fontsize=14, fontweight='bold', y=0.98)

for ax, y_true, feat_styles, title in zip(axes,
        [y1, y2],
        [feat_styles_t1, feat_styles_t2],
        ['Type1 PR', 'Type2 PR']):
    baseline = y_true.mean()
    ax.axhline(baseline, color='gray', lw=1.5, linestyle='--',
               label=f'Baseline (prevalence={baseline:.3f})')
    for scores, ls, color, label in feat_styles:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = sk_auc(rec, prec)
        ax.plot(rec, prec, linestyle=ls, color=color, lw=2,
                label=f'{label.split("(")[0].strip()} (AUC={pr_auc:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3); ax.set_facecolor('#f8f8f8')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's12_fig3_pr_curves.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s12_fig3_pr_curves.png")

# ── Figure 4: Performance summary table ──────────────────────────────────
print("Figure 4: Performance summary...")
res_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("Step 12: Rule Performance Summary",
             fontsize=14, fontweight='bold', y=0.98)

x = np.arange(len(res_df))
w = 0.25
bars1 = ax.bar(x - w, res_df['precision'], w, label='Precision',
               color='#1565c0', alpha=0.8)
bars2 = ax.bar(x,      res_df['recall'],    w, label='Recall',
               color='#2e7d32', alpha=0.8)
bars3 = ax.bar(x + w,  res_df['f1'],        w, label='F1',
               color='#c62828', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(res_df['label'], rotation=25, ha='right', fontsize=7)
ax.set_ylabel('Score')
ax.set_ylim([0, 1.1])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#f8f8f8')
ax.axhline(0.8, color='gray', lw=1, linestyle=':', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's12_fig4_performance_summary.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s12_fig4_performance_summary.png")

# ── Save thresholds ────────────────────────────────────────────────────────
thresholds = {
    'type1': {
        'mean_abs_div_threshold': t1a,
        'slope_dev_threshold':    t1b,
        'val_rmse_zscore_threshold': t1c,
    },
    'type2': {
        'mono_rate_threshold':  -t2a,
        'val_rmse_threshold':   -t2b,
        'max_jump_threshold':   -t2c,
    }
}
with open(os.path.join(RES, 'rule_thresholds.json'), 'w') as f:
    json.dump(thresholds, f, indent=2)

res_df.to_csv(os.path.join(RES, 'rule_performance.csv'), index=False)
print(f"\n  Saved: rule_thresholds.json, rule_performance.csv")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 12 COMPLETE -- KEY TAKEAWAYS")
print("="*65)
print("\n  TYPE1 RULES:")
print(f"    Best single rule: mean_abs_div >= {t1a:.1f}")
t1_best = max([r for r in results if 'Type1' not in r['label'] and 'R1' in r['label']
               or 'R1' in r.get('label', '')],
              key=lambda x: x['f1'], default=None)
for r in results:
    if 'R1' in r['label']:
        print(f"    {r['label']:<42} F1={r['f1']:.3f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f}")
print("\n  TYPE2 RULES:")
for r in results:
    if 'R2' in r['label']:
        print(f"    {r['label']:<42} F1={r['f1']:.3f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f}")
print("\n  COMBINED:")
for r in results:
    if 'R2' not in r['label'] and 'R1' not in r['label']:
        print(f"    {r['label']:<42} F1={r['f1']:.3f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f}")
print(f"\n  -> Next: Step 13 -- Meta-classifier (LR + XGBoost + SHAP)")
print(f"  -> Output files in: {RES}/")
