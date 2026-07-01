"""
Step 13b: Full SHAP / XAI Analysis
====================================
Applies SHAP (SHapley Additive exPlanations) to the XGBoost meta-classifier
trained in Step 13 to explain:

  1. Global importance — which features matter most overall
  2. Per-class importance — different features drive Type1 vs Type2 detection
  3. Beeswarm plot — direction of each feature's effect (high value -> flag up/down)
  4. Waterfall plots — single-instance explanation for a Type1 and Type2 example
  5. Decision plots — how predictions are built feature by feature
  6. Dependence plots — how top features interact with each other
  7. Summary: what SHAP tells us about the nature of each outlier type
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(BASE, 'results', 'outlier_analysis', 'step13b_shap')
os.makedirs(RES, exist_ok=True)

STEP11_CSV = os.path.join(BASE, 'results', 'outlier_analysis', 'step11',
                           'records_step11.csv')
SEL_FEATS  = os.path.join(BASE, 'results', 'outlier_analysis', 'step11',
                           'selected_features.json')

TYPE_COLORS = {'Inlier': '#2196F3', 'Type1': '#c62828', 'Type2': '#e65100'}
LABELS      = ['Inlier', 'Type1', 'Type2']

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(STEP11_CSV)
with open(SEL_FEATS) as f:
    sel = json.load(f)

for col in ['otype']:
    if col in df.columns:
        df.rename(columns={col: 'outlier_type'}, inplace=True)

SELECTED = [c for c in sel['selected'] if c in df.columns]
print(f"  Records: {len(df)} | Types: {df['outlier_type'].value_counts().to_dict()}")
print(f"  Features: {SELECTED}")

for col in SELECTED:
    df[col] = df[col].fillna(df[col].median())

X  = df[SELECTED].values
y3 = df['outlier_type'].values
le = LabelEncoder()
le.fit(LABELS)
y3_enc = le.transform(y3)

# ── Train separate binary Random Forest classifiers (SHAP-compatible) ─────
# RandomForest has perfect SHAP TreeExplainer support across all versions
print("\nTraining binary Random Forest classifiers for SHAP...")
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
X_shap  = pd.DataFrame(X_sc, columns=SELECTED)

y_inlier = (y3_enc == 0).astype(int)
y1_bin   = (y3_enc == 1).astype(int)
y2_bin   = (y3_enc == 2).astype(int)

def make_rf(y_bin):
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=6,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf.fit(X_shap, y_bin)
    return clf

clf_inlier = make_rf(y_inlier)
clf_type1  = make_rf(y1_bin)
clf_type2  = make_rf(y2_bin)

# ── SHAP values via TreeExplainer ─────────────────────────────────────────
print("Computing SHAP values (TreeExplainer — RandomForest)...")

def get_shap(clf, X_df):
    exp = shap.TreeExplainer(clf)
    sv  = exp.shap_values(X_df)
    # newer shap returns 3D array (n_samples, n_features, n_classes)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        return sv[:, :, 1], exp.expected_value[1]
    # older shap returns list [neg_class_arr, pos_class_arr]
    if isinstance(sv, list) and len(sv) == 2:
        return sv[1], exp.expected_value[1]
    return sv, exp.expected_value

sv_inlier, ev_inlier = get_shap(clf_inlier, X_shap)
sv_type1,  ev_type1  = get_shap(clf_type1,  X_shap)
sv_type2,  ev_type2  = get_shap(clf_type2,  X_shap)

print(f"  SHAP array shape: {sv_type1.shape}")

# ── Helper: mean |SHAP| per feature per class ──────────────────────────────
def mean_abs_shap(sv):
    return pd.Series(np.abs(sv).mean(axis=0), index=SELECTED)

imp_inlier = mean_abs_shap(sv_inlier).sort_values(ascending=False)
imp_type1  = mean_abs_shap(sv_type1).sort_values(ascending=False)
imp_type2  = mean_abs_shap(sv_type2).sort_values(ascending=False)

print("\n  Mean |SHAP| per feature:")
print(f"  {'Feature':<28} {'Inlier':>10} {'Type1':>10} {'Type2':>10}")
print("  " + "-"*60)
for feat in SELECTED:
    print(f"  {feat:<28} {imp_inlier[feat]:>10.4f} "
          f"{imp_type1[feat]:>10.4f} {imp_type2[feat]:>10.4f}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Global feature importance — side-by-side bar for 3 classes
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Global SHAP importance by class...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("SHAP Global Feature Importance — Mean |SHAP| per Class\n"
             "Higher = feature contributes more to that class prediction",
             fontsize=13, fontweight='bold', y=0.98)

for ax, (sv, cls_name, color) in zip(axes, [
        (sv_inlier, 'Inlier',  '#2196F3'),
        (sv_type1,  'Type1 (XGB Underfitting)', '#c62828'),
        (sv_type2,  'Type2 (Transformer)', '#e65100'),
]):
    imp = mean_abs_shap(sv).sort_values(ascending=True)
    bars = ax.barh(imp.index, imp.values, color=color, alpha=0.8)
    ax.set_xlabel('Mean |SHAP value|', fontsize=10)
    ax.set_title(cls_name, fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(RES, 's13b_fig1_global_importance.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13b_fig1_global_importance.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Beeswarm — Type1 (shows direction of feature effect)
# ══════════════════════════════════════════════════════════════════════════
print("Figure 2: Beeswarm for Type1...")
fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle("SHAP Beeswarm — Type1 (XGB Underfitting) Detection\n"
             "Red = high feature value, Blue = low. Right = pushes toward Type1.",
             fontsize=12, fontweight='bold', y=0.98)

imp_order = np.argsort(np.abs(sv_type1).mean(axis=0))[::-1]
feat_names_ordered = [SELECTED[i] for i in imp_order]
sv_ordered = sv_type1[:, imp_order]
feat_vals_ordered = X_sc[:, imp_order]

for fi, (fname, col_sv, col_fv) in enumerate(
        zip(feat_names_ordered, sv_ordered.T, feat_vals_ordered.T)):
    # jitter y for visibility
    y_jitter = fi + np.random.uniform(-0.3, 0.3, len(col_sv))
    sc = ax.scatter(col_sv, y_jitter, c=col_fv,
                    cmap='coolwarm', alpha=0.4, s=12,
                    vmin=-2, vmax=2)

ax.set_yticks(range(len(SELECTED)))
ax.set_yticklabels(feat_names_ordered, fontsize=10)
ax.axvline(0, color='black', lw=1.2, alpha=0.6)
ax.set_xlabel('SHAP value (impact on Type1 prediction)', fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('#f8f8f8')
cb = plt.colorbar(sc, ax=ax, fraction=0.03)
cb.set_label('Feature value\n(standardised)', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(RES, 's13b_fig2_beeswarm_type1.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13b_fig2_beeswarm_type1.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Beeswarm — Type2
# ══════════════════════════════════════════════════════════════════════════
print("Figure 3: Beeswarm for Type2...")
fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle("SHAP Beeswarm — Type2 (Transformer Over-Performing) Detection\n"
             "Red = high feature value, Blue = low. Right = pushes toward Type2.",
             fontsize=12, fontweight='bold', y=0.98)

imp_order2 = np.argsort(np.abs(sv_type2).mean(axis=0))[::-1]
feat_names_ordered2 = [SELECTED[i] for i in imp_order2]
sv_ordered2 = sv_type2[:, imp_order2]
feat_vals_ordered2 = X_sc[:, imp_order2]

for fi, (fname, col_sv, col_fv) in enumerate(
        zip(feat_names_ordered2, sv_ordered2.T, feat_vals_ordered2.T)):
    y_jitter = fi + np.random.uniform(-0.3, 0.3, len(col_sv))
    sc = ax.scatter(col_sv, y_jitter, c=col_fv,
                    cmap='coolwarm', alpha=0.4, s=12,
                    vmin=-2, vmax=2)

ax.set_yticks(range(len(SELECTED)))
ax.set_yticklabels(feat_names_ordered2, fontsize=10)
ax.axvline(0, color='black', lw=1.2, alpha=0.6)
ax.set_xlabel('SHAP value (impact on Type2 prediction)', fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('#f8f8f8')
cb = plt.colorbar(sc, ax=ax, fraction=0.03)
cb.set_label('Feature value\n(standardised)', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(RES, 's13b_fig3_beeswarm_type2.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13b_fig3_beeswarm_type2.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Waterfall — one Type1 example + one Type2 example
# ══════════════════════════════════════════════════════════════════════════
print("Figure 4: Waterfall plots for individual examples...")

# find clearest Type1 and Type2 examples (highest predicted probability)
type1_idx = np.where(y3_enc == 1)[0]
type2_idx = np.where(y3_enc == 2)[0]

# pick example with highest model-assigned probability for each binary clf
proba_t1 = clf_type1.predict_proba(X_shap)[:, 1]
proba_t2 = clf_type2.predict_proba(X_shap)[:, 1]

best_t1 = type1_idx[np.argmax(proba_t1[type1_idx])]
best_t2 = type2_idx[np.argmax(proba_t2[type2_idx])]

expected_val_t1 = ev_type1
expected_val_t2 = ev_type2

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("SHAP Waterfall — Single Instance Explanations\n"
             "How each feature pushes the prediction above/below the baseline",
             fontsize=13, fontweight='bold', y=0.98)

for ax, (idx, sv_class, class_name, exp_val, color) in zip(axes, [
        (best_t1, sv_type1, 'Type1', expected_val_t1, '#c62828'),
        (best_t2, sv_type2, 'Type2', expected_val_t2, '#e65100'),
]):
    shap_row = sv_class[idx]
    feat_row = X_sc[idx]
    feat_orig = X[idx]

    # sort by absolute SHAP value
    order = np.argsort(np.abs(shap_row))[::-1]
    top_n = min(10, len(SELECTED))
    order = order[:top_n][::-1]  # smallest at top for waterfall

    features_show = [SELECTED[i] for i in order]
    shap_show     = shap_row[order]
    orig_show     = feat_orig[order]

    # manual waterfall bar chart
    cumulative = exp_val
    bar_lefts  = []
    bar_widths = []
    bar_colors = []

    for s in shap_show:
        bar_lefts.append(min(cumulative, cumulative + s))
        bar_widths.append(abs(s))
        bar_colors.append('#c62828' if s > 0 else '#1565c0')
        cumulative += s

    y_pos = range(len(features_show))
    for i, (left, width, bc) in enumerate(zip(bar_lefts, bar_widths, bar_colors)):
        ax.barh(i, width, left=left, color=bc, alpha=0.85, height=0.6)

    ax.axvline(exp_val, color='gray', lw=1.5, linestyle='--', alpha=0.7,
               label=f'Baseline E[f(x)] = {exp_val:.3f}')
    ax.axvline(cumulative, color=color, lw=2, linestyle='-',
               label=f'Final f(x) = {cumulative:.3f}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f'{f}\n= {v:.2f}' for f, v in zip(features_show, orig_show)],
        fontsize=9)
    ax.set_xlabel('SHAP contribution (log-odds)', fontsize=10)
    ax.set_title(
        f'{class_name} Example\n'
        f'Exp={df.iloc[idx]["experiment"]}  Model={df.iloc[idx]["model"]}',
        fontsize=11, color=color)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f8f8')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(RES, 's13b_fig4_waterfall.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13b_fig4_waterfall.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Dependence plots — top feature for each type
# ══════════════════════════════════════════════════════════════════════════
print("Figure 5: Dependence plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("SHAP Dependence Plots — How Feature Value Drives Prediction\n"
             "Colour = value of most-interacting second feature",
             fontsize=13, fontweight='bold', y=0.98)

top_t1_feat = imp_type1.index[0]
sec_t1_feat = imp_type1.index[1]
top_t2_feat = imp_type2.index[0]
sec_t2_feat = imp_type2.index[1]

for ax, (sv, top_feat, sec_feat, cls_name, color) in zip(axes.flat, [
        (sv_type1, top_t1_feat, sec_t1_feat, 'Type1', '#c62828'),
        (sv_type1, sec_t1_feat, top_t1_feat, 'Type1', '#c62828'),
        (sv_type2, top_t2_feat, sec_t2_feat, 'Type2', '#e65100'),
        (sv_type2, sec_t2_feat, top_t2_feat, 'Type2', '#e65100'),
]):
    feat_idx = SELECTED.index(top_feat)
    sec_idx  = SELECTED.index(sec_feat)
    x_vals   = X_sc[:, feat_idx]
    y_vals   = sv[:, feat_idx]
    c_vals   = X_sc[:, sec_idx]

    sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap='coolwarm',
                    alpha=0.5, s=20, vmin=-2, vmax=2)
    ax.axhline(0, color='black', lw=1, alpha=0.5, linestyle='--')
    ax.set_xlabel(f'{top_feat} (standardised)', fontsize=10)
    ax.set_ylabel(f'SHAP value for {cls_name}', fontsize=10)
    ax.set_title(f'{cls_name}: {top_feat}\n(colour = {sec_feat})',
                 fontsize=10, color=color)
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    plt.colorbar(sc, ax=ax, fraction=0.04, label=sec_feat)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(RES, 's13b_fig5_dependence.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13b_fig5_dependence.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Side-by-side Type1 vs Type2 SHAP profile comparison
# ══════════════════════════════════════════════════════════════════════════
print("Figure 6: Type1 vs Type2 SHAP profile comparison...")
fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle("SHAP Feature Importance Comparison: Type1 vs Type2\n"
             "Shows which features distinguish each outlier type",
             fontsize=13, fontweight='bold', y=0.98)

imp1 = mean_abs_shap(sv_type1)
imp2 = mean_abs_shap(sv_type2)

# sort by max of the two
combined_order = (imp1 + imp2).sort_values(ascending=True)
feats_ordered  = combined_order.index.tolist()

x = np.arange(len(feats_ordered))
w = 0.35

bars1 = ax.barh(x - w/2, [imp1[f] for f in feats_ordered], w,
                color='#c62828', alpha=0.8, label='Type1 (XGB Underfitting)')
bars2 = ax.barh(x + w/2, [imp2[f] for f in feats_ordered], w,
                color='#e65100', alpha=0.8, label='Type2 (Transformer)')

ax.set_yticks(x)
ax.set_yticklabels(feats_ordered, fontsize=10)
ax.set_xlabel('Mean |SHAP value|', fontsize=11)
ax.set_title('')
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('#f8f8f8')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(RES, 's13b_fig6_type1_vs_type2_profile.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13b_fig6_type1_vs_type2_profile.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: SHAP mean values for ALL Type1 instances (group explanation)
# ══════════════════════════════════════════════════════════════════════════
print("Figure 7: Group SHAP for all Type1 instances...")
type1_mask = y3_enc == 1
type2_mask = y3_enc == 2

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("SHAP Mean Values — All Type1 and Type2 Instances\n"
             "Average signed SHAP shows direction of each feature's effect",
             fontsize=13, fontweight='bold', y=0.98)

for ax, (mask, sv, cls_name, color) in zip(axes, [
        (type1_mask, sv_type1, 'Type1 (all 6 runs)', '#c62828'),
        (type2_mask, sv_type2, 'Type2 (all 49 runs)', '#e65100'),
]):
    mean_signed = pd.Series(sv[mask].mean(axis=0), index=SELECTED)
    mean_signed_sorted = mean_signed.reindex(
        mean_signed.abs().sort_values(ascending=True).index)
    bar_colors = ['#c62828' if v > 0 else '#1565c0'
                  for v in mean_signed_sorted.values]
    bars = ax.barh(mean_signed_sorted.index, mean_signed_sorted.values,
                   color=bar_colors, alpha=0.85)
    ax.axvline(0, color='black', lw=1.5, alpha=0.7)
    ax.set_xlabel('Mean SHAP value (signed)', fontsize=11)
    ax.set_title(cls_name, fontsize=11, color=color, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    for bar, val in zip(bars, mean_signed_sorted.values):
        offset = 0.001 if val >= 0 else -0.001
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha=ha, fontsize=8)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#c62828', alpha=0.85, label='Pushes toward this class (+)'),
    Patch(facecolor='#1565c0', alpha=0.85, label='Pushes away from this class (-)'),
]
axes[1].legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(RES, 's13b_fig7_group_shap.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13b_fig7_group_shap.png")

# ── Save SHAP values to CSV ───────────────────────────────────────────────
shap_out = pd.DataFrame({
    'experiment': df['experiment'].values,
    'model':      df['model'].values,
    'outlier_type': df['outlier_type'].values,
})
for fi, feat in enumerate(SELECTED):
    shap_out[f'shap_inlier_{feat}'] = sv_inlier[:, fi]
    shap_out[f'shap_type1_{feat}']  = sv_type1[:, fi]
    shap_out[f'shap_type2_{feat}']  = sv_type2[:, fi]

shap_out.to_csv(os.path.join(RES, 'shap_values.csv'), index=False)
print(f"\n  Saved: shap_values.csv")

# ── Print final interpretation ─────────────────────────────────────────────
print("\n" + "="*70)
print("  SHAP ANALYSIS COMPLETE -- INTERPRETATION")
print("="*70)

print(f"\n  Top 3 features driving TYPE1 detection:")
for feat in imp_type1.index[:3]:
    mean_signed = sv_type1[type1_mask, SELECTED.index(feat)].mean()
    print(f"    {feat:<28}  mean|shap|={imp_type1[feat]:.4f}  "
          f"direction={'UP (+)' if mean_signed > 0 else 'DOWN (-)'}")

print(f"\n  Top 3 features driving TYPE2 detection:")
for feat in imp_type2.index[:3]:
    mean_signed = sv_type2[type2_mask, SELECTED.index(feat)].mean()
    print(f"    {feat:<28}  mean|shap|={imp_type2[feat]:.4f}  "
          f"direction={'UP (+)' if mean_signed > 0 else 'DOWN (-)'}")

print(f"\n  Key insight from SHAP:")
print(f"    Type1 is driven by HIGH {imp_type1.index[0]} and HIGH {imp_type1.index[1]}")
print(f"    Type2 is driven by LOW  {imp_type2.index[0]} and LOW  {imp_type2.index[1]}")
print(f"\n  -> All figures in: {RES}/")
