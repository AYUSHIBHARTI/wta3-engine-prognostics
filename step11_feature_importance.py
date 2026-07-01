"""
Step 11: Feature Importance Ranking — Meta-Classifier Input Selection
======================================================================
Goal: Aggregate ALL features collected across Steps 3–10 and rank them
by their ability to predict outlier status. This tells us:
  (a) Which single features are the strongest individual signals
  (b) Which features are redundant (highly correlated)
  (c) Which features should be inputs to the meta-classifier in Step 13

Features collected:
  Step 3: val_rmse_zscore, gap_ratio, gap_zscore
  Step 4: mono_violations, mono_rate, max_jump, mean_abs_diff,
          mean_rolling_var, mean_ma_dev, slope, slope_dev,
          slope_consistency, final_pred
  Step 9: mean_corr_with_others, min_corr_with_others, rank_in_experiment
  Step 10: mean_abs_div, max_abs_div, div_growth_ratio, div_sign_cons,
           early_div, late_div

Analysis:
  1. ROC-AUC for each feature individually (Type1 and Type2)
  2. Feature correlation heatmap (identify redundancy)
  3. Random Forest feature importance (from RF meta-classifier)
  4. Select top-K non-redundant features for Step 13
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(BASE, 'results', 'outlier_analysis', 'step11')
os.makedirs(RES, exist_ok=True)

STEP3_CSV  = os.path.join(BASE, 'results', 'outlier_analysis', 'step3',
                           'records_with_dynamics.csv')
STEP4_CSV  = os.path.join(BASE, 'results', 'outlier_analysis', 'step4',
                           'records_with_trajectory.csv')
STEP9_CSV  = os.path.join(BASE, 'results', 'outlier_analysis', 'step9',
                           'records_step9.csv')
STEP10_CSV = os.path.join(BASE, 'results', 'outlier_analysis', 'step10',
                           'records_step10.csv')

TYPE_COLORS = {'Inlier': '#2196F3', 'Type1': '#c62828', 'Type2': '#e65100'}

# ── load and merge ─────────────────────────────────────────────────────────
print("Loading and merging feature tables...")

df3  = pd.read_csv(STEP3_CSV)
df4  = pd.read_csv(STEP4_CSV)
df9  = pd.read_csv(STEP9_CSV)
df10 = pd.read_csv(STEP10_CSV)

# check available columns
print("  Step3 cols:", [c for c in df3.columns if c not in ['experiment','model']])
print("  Step4 cols:", [c for c in df4.columns if c not in ['experiment','model']])
print("  Step9 cols:", [c for c in df9.columns if c not in ['experiment','model']])
print("  Step10 cols:", [c for c in df10.columns if c not in ['experiment','model']])

# normalise outlier_type column
def get_otype_col(d):
    for col in ['outlier_type', 'otype']:
        if col in d.columns:
            return col
    return None

ot3  = get_otype_col(df3)
ot4  = get_otype_col(df4)
ot9  = get_otype_col(df9)
ot10 = get_otype_col(df10)

# rename to common name
for d, col in [(df3, ot3), (df4, ot4), (df9, ot9), (df10, ot10)]:
    if col and col != 'outlier_type':
        d.rename(columns={col: 'outlier_type'}, inplace=True)

# merge on experiment + model
base = df3[['experiment','model','outlier_type']].copy()
merge_dfs = [
    (df3,  ['val_rmse_zscore','gap_ratio','gap_zscore'] if 'val_rmse_zscore' in df3.columns else []),
    (df4,  ['mono_violations','mono_rate','max_jump','mean_abs_diff',
             'mean_rolling_var','mean_ma_dev','slope','slope_dev',
             'slope_consistency','final_pred']),
    (df9,  ['mean_corr_with_others','min_corr_with_others','rank_in_experiment']),
    (df10, ['mean_abs_div','max_abs_div','div_growth_ratio','div_sign_cons',
            'early_div','late_div']),
]

merged = base.copy()
for src_df, feat_cols in merge_dfs:
    existing_cols = [c for c in feat_cols if c in src_df.columns]
    if existing_cols:
        merged = merged.merge(src_df[['experiment','model'] + existing_cols],
                              on=['experiment','model'], how='left')

# also get val_rmse directly from step4 (always present)
if 'val_rmse' in df4.columns:
    merged = merged.merge(df4[['experiment','model','val_rmse']],
                          on=['experiment','model'], how='left')

print(f"\n  Merged: {len(merged)} rows, {len(merged.columns)} columns")
print(f"  Outlier counts: {merged['outlier_type'].value_counts().to_dict()}")

# ── define feature list ────────────────────────────────────────────────────
FEATURE_COLS = [c for c in merged.columns
                if c not in ['experiment','model','outlier_type']
                and merged[c].dtype in [np.float64, np.int64, float, int]]
print(f"\n  Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── ROC-AUC for each feature ───────────────────────────────────────────────
print("\n" + "="*70)
print("  INDIVIDUAL FEATURE ROC-AUC")
print("="*70)
print(f"\n{'Feature':<28} {'Type1 AUC':>12} {'Type2 AUC':>12}  {'Combined'}")
print("-"*65)

y1 = (merged['outlier_type'] == 'Type1').astype(int)
y2 = (merged['outlier_type'] == 'Type2').astype(int)

auc_records = []
for feat in FEATURE_COLS:
    scores = merged[feat].fillna(merged[feat].median())
    try:
        fpr1, tpr1, _ = roc_curve(y1, scores)
        a1 = sk_auc(fpr1, tpr1)
        # ensure AUC >= 0.5 (flip direction if needed)
        if a1 < 0.5:
            a1 = 1.0 - a1
            scores_inv = -scores
            fpr1, tpr1, _ = roc_curve(y1, scores_inv)
        fpr2, tpr2, _ = roc_curve(y2, scores)
        a2 = sk_auc(fpr2, tpr2)
        if a2 < 0.5:
            a2 = 1.0 - a2
        combined = (a1 + a2) / 2
        auc_records.append(dict(feature=feat, auc_type1=a1, auc_type2=a2,
                                 combined_auc=combined))
        print(f"  {feat:<26} {a1:>12.4f} {a2:>12.4f}  {combined:.4f}")
    except Exception as e:
        print(f"  {feat:<26}  ERROR: {e}")

auc_df = pd.DataFrame(auc_records).sort_values('combined_auc', ascending=False)

# ── Feature correlation matrix ─────────────────────────────────────────────
print("\nComputing feature correlation matrix...")
feat_data = merged[FEATURE_COLS].fillna(merged[FEATURE_COLS].median())
corr_matrix = feat_data.corr(method='spearman')

# ── Random Forest feature importance ──────────────────────────────────────
print("Computing Random Forest feature importance...")
X = feat_data.values
y_any = ((merged['outlier_type'] == 'Type1') |
         (merged['outlier_type'] == 'Type2')).astype(int)

rf = RandomForestClassifier(n_estimators=200, random_state=42,
                              class_weight='balanced')
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
rf.fit(X_sc, y_any)
importances = rf.feature_importances_

# also RF separately for Type1 and Type2
rf1 = RandomForestClassifier(n_estimators=200, random_state=42,
                               class_weight='balanced')
rf1.fit(X_sc, y1)
imp1 = rf1.feature_importances_

rf2 = RandomForestClassifier(n_estimators=200, random_state=42,
                               class_weight='balanced')
rf2.fit(X_sc, y2)
imp2 = rf2.feature_importances_

importance_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'rf_any':  importances,
    'rf_type1':imp1,
    'rf_type2':imp2,
}).sort_values('rf_any', ascending=False)

print(f"\n  Top-10 by RF importance (any outlier):")
for _, row in importance_df.head(10).iterrows():
    print(f"    {row['feature']:<28} any={row['rf_any']:.4f}  "
          f"t1={row['rf_type1']:.4f}  t2={row['rf_type2']:.4f}")

# ── Select non-redundant top features ────────────────────────────────────
print("\nSelecting non-redundant top features (corr threshold = 0.85)...")
sorted_feats = auc_df['feature'].tolist()
selected = []
for feat in sorted_feats:
    if not selected:
        selected.append(feat)
        continue
    max_corr = max(abs(corr_matrix.loc[feat, s]) for s in selected
                   if s in corr_matrix.columns)
    if max_corr < 0.85:
        selected.append(feat)
    if len(selected) >= 12:
        break

print(f"  Selected {len(selected)} features: {selected}")

# ── Figure 1: AUC bar chart ───────────────────────────────────────────────
print("\nFigure 1: AUC bar chart...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Step 11: Individual Feature ROC-AUC for Outlier Detection",
             fontsize=14, fontweight='bold', y=0.98)

for ax, (col, title, color) in zip(axes, [
        ('auc_type1', 'Type1 (XGB Underfitting) Detection', '#c62828'),
        ('auc_type2', 'Type2 (Transformer Divergent) Detection', '#e65100'),
]):
    sorted_df = auc_df.sort_values(col, ascending=True)
    bars = ax.barh(sorted_df['feature'], sorted_df[col],
                   color=color, alpha=0.7)
    ax.axvline(0.5, color='black', lw=1.5, linestyle='--', alpha=0.6,
               label='Random (0.50)')
    ax.axvline(0.8, color='green', lw=1.5, linestyle=':', alpha=0.6,
               label='Good (0.80)')
    ax.set_xlabel('ROC-AUC', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    ax.set_xlim([0.3, 1.05])
    for bar, val in zip(bars, sorted_df[col]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's11_fig1_auc_barchart.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s11_fig1_auc_barchart.png")

# ── Figure 2: Feature correlation heatmap ────────────────────────────────
print("Figure 2: Feature correlation heatmap...")
fig, ax = plt.subplots(figsize=(14, 12))
fig.suptitle("Step 11: Spearman Correlation Between Features\n"
             "(High correlation = redundant features)",
             fontsize=13, fontweight='bold', y=0.98)

im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(FEATURE_COLS)))
ax.set_yticks(range(len(FEATURE_COLS)))
ax.set_xticklabels(FEATURE_COLS, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(FEATURE_COLS, fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.03, label='Spearman r')
for i in range(len(FEATURE_COLS)):
    for j in range(len(FEATURE_COLS)):
        v = corr_matrix.values[i, j]
        txt_color = 'black' if abs(v) < 0.6 else 'white'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=6, color=txt_color)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(RES, 's11_fig2_feature_correlation.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s11_fig2_feature_correlation.png")

# ── Figure 3: RF feature importance ───────────────────────────────────────
print("Figure 3: RF feature importance...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Step 11: Random Forest Feature Importances",
             fontsize=14, fontweight='bold', y=0.98)

for ax, (col, title, color) in zip(axes, [
        ('rf_any',   'Any Outlier',             '#6a1b9a'),
        ('rf_type1', 'Type1 (XGB) Only',        '#c62828'),
        ('rf_type2', 'Type2 (Transformer) Only','#e65100'),
]):
    sorted_imp = importance_df.sort_values(col, ascending=True)
    bars = ax.barh(sorted_imp['feature'], sorted_imp[col],
                   color=color, alpha=0.7)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    for bar, val in zip(bars, sorted_imp[col]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's11_fig3_rf_importance.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s11_fig3_rf_importance.png")

# ── Figure 4: Selected features scatter matrix ───────────────────────────
print("Figure 4: Selected features summary chart...")
top5_t1 = auc_df.nlargest(5, 'auc_type1')['feature'].tolist()
top5_t2 = auc_df.nlargest(5, 'auc_type2')['feature'].tolist()
top_feats_show = list(dict.fromkeys(top5_t1 + top5_t2))[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Step 11: Top Feature Distributions by Outlier Type",
             fontsize=14, fontweight='bold', y=0.98)

for ax, feat in zip(axes.flat, top_feats_show):
    data = [merged[merged['outlier_type'] == t][feat].dropna().values
            for t in ['Inlier', 'Type1', 'Type2']]
    bp = ax.boxplot(data, tick_labels=['Inlier', 'Type1', 'Type2'],
                    patch_artist=True, notch=False, showfliers=True)
    for patch, color in zip(bp['boxes'],
                             [TYPE_COLORS[t] for t in ['Inlier','Type1','Type2']]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    row_auc = auc_df[auc_df['feature'] == feat]
    if len(row_auc):
        a1 = row_auc['auc_type1'].values[0]
        a2 = row_auc['auc_type2'].values[0]
        ax.set_title(f'{feat}\n(T1 AUC={a1:.3f}, T2 AUC={a2:.3f})', fontsize=9)
    else:
        ax.set_title(feat, fontsize=9)
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f8f8f8')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's11_fig4_top_feature_distributions.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s11_fig4_top_feature_distributions.png")

# ── Save outputs ───────────────────────────────────────────────────────────
auc_df.to_csv(os.path.join(RES, 'feature_auc_ranking.csv'), index=False)
importance_df.to_csv(os.path.join(RES, 'feature_rf_importance.csv'), index=False)
merged.to_csv(os.path.join(RES, 'records_step11.csv'), index=False)

# save selected feature list for step 13
with open(os.path.join(RES, 'selected_features.json'), 'w') as f:
    json.dump({'selected': selected, 'all_features': FEATURE_COLS,
               'top5_type1': top5_t1, 'top5_type2': top5_t2}, f, indent=2)

print(f"\n  Saved: feature_auc_ranking.csv, feature_rf_importance.csv, "
      f"records_step11.csv, selected_features.json")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 11 COMPLETE -- KEY TAKEAWAYS")
print("="*65)
print("\n  Top 5 features for Type1 detection:")
for _, r in auc_df.nlargest(5, 'auc_type1').iterrows():
    print(f"    {r['feature']:<28}  AUC={r['auc_type1']:.4f}")
print("\n  Top 5 features for Type2 detection:")
for _, r in auc_df.nlargest(5, 'auc_type2').iterrows():
    print(f"    {r['feature']:<28}  AUC={r['auc_type2']:.4f}")
print(f"\n  Selected non-redundant features for Step 13: {selected}")
print(f"\n  -> Next: Step 12 -- Simple rule-based single-model indicator")
print(f"  -> Output files in: {RES}/")
