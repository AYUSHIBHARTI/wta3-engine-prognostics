"""
Step 13: Meta-Classifier — Logistic Regression + XGBoost + SHAP
================================================================
Goal: Train a supervised classifier that predicts outlier status
(Inlier / Type1 / Type2) from the 12 non-redundant features selected
in Step 11. This is the single-model outlier detector answering the
professor's research question.

Two classifiers:
  1. Logistic Regression (interpretable, strong baseline)
  2. XGBoost classifier (non-linear, likely better performance)

Evaluation:
  - Stratified K-Fold cross-validation (k=5) — no data leakage
  - Macro and weighted F1, Precision, Recall
  - ROC-AUC per class (OvR)
  - Confusion matrix
  - SHAP values for XGBoost to explain feature contributions

Binary variants:
  - Type1 vs Rest (rare positive — underfitting XGB)
  - Type2 vs Rest (more common positive — over-performing Transformer)
  - Any Outlier vs Inlier

Output: full classification report + SHAP plots for thesis
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      cross_val_predict)
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, auc as sk_auc,
                              f1_score)
import xgboost as xgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  [WARNING] shap not installed — SHAP plots will be skipped")

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(BASE, 'results', 'outlier_analysis', 'step13')
os.makedirs(RES, exist_ok=True)

STEP11_CSV = os.path.join(BASE, 'results', 'outlier_analysis', 'step11',
                           'records_step11.csv')
SEL_FEATS  = os.path.join(BASE, 'results', 'outlier_analysis', 'step11',
                           'selected_features.json')

TYPE_COLORS = {'Inlier': '#2196F3', 'Type1': '#c62828', 'Type2': '#e65100'}
LABELS = ['Inlier', 'Type1', 'Type2']

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(STEP11_CSV)
with open(SEL_FEATS) as f:
    sel = json.load(f)

# normalise column
for col in ['otype', 'outlier_type']:
    if col in df.columns and col != 'outlier_type':
        df.rename(columns={col: 'outlier_type'}, inplace=True)

print(f"  Records: {len(df)}, types: {df['outlier_type'].value_counts().to_dict()}")

SELECTED = [c for c in sel['selected'] if c in df.columns]
print(f"  Selected features ({len(SELECTED)}): {SELECTED}")

# impute
for col in SELECTED:
    df[col] = df[col].fillna(df[col].median())

X    = df[SELECTED].values
y3   = df['outlier_type'].values                        # 3-class
y_any = (df['outlier_type'] != 'Inlier').astype(int)   # binary any
y1   = (df['outlier_type'] == 'Type1').astype(int)
y2   = (df['outlier_type'] == 'Type2').astype(int)

le = LabelEncoder()
le.fit(LABELS)
y3_enc = le.transform(y3)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Model 1: Logistic Regression ──────────────────────────────────────────
print("\n" + "="*70)
print("  LOGISTIC REGRESSION")
print("="*70)

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                                random_state=42, multi_class='ovr'))
])

# 3-class
lr_preds3 = cross_val_predict(lr_pipe, X, y3, cv=cv, method='predict')
lr_proba3 = cross_val_predict(lr_pipe, X, y3, cv=cv, method='predict_proba')
lr_pipe.fit(X, y3)

print("\n  3-Class Classification Report (Inlier / Type1 / Type2):")
print(classification_report(y3, lr_preds3, target_names=LABELS))

# binary
for y_bin, label in [(y_any, 'Any Outlier'), (y1, 'Type1'), (y2, 'Type2')]:
    preds = cross_val_predict(lr_pipe, X, y_bin, cv=cv, method='predict')
    f1 = f1_score(y_bin, preds, average='binary', zero_division=0)
    print(f"  LR {label:<15} binary F1 = {f1:.4f}")

# ── Model 2: XGBoost ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("  XGBOOST CLASSIFIER")
print("="*70)

n_per_class = np.bincount(y3_enc)
scale_pos = n_per_class[0] / np.maximum(n_per_class, 1)  # inverse freq
sample_weights = np.array([scale_pos[c] for c in y3_enc])

xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, verbosity=0,
    num_class=3, objective='multi:softprob',
)
scaler_xgb = StandardScaler()
X_sc = scaler_xgb.fit_transform(X)

# cross-validate
xgb_preds3 = cross_val_predict(xgb_clf, X_sc, y3_enc, cv=cv, method='predict')
xgb_proba3 = cross_val_predict(xgb_clf, X_sc, y3_enc, cv=cv, method='predict_proba')

xgb_preds_labels = le.inverse_transform(xgb_preds3)
print("\n  3-Class Classification Report (XGBoost):")
print(classification_report(y3, xgb_preds_labels, target_names=LABELS))

# Binary tasks
for y_bin, label in [(y_any, 'Any Outlier'), (y1, 'Type1'), (y2, 'Type2')]:
    xgb_bin = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, use_label_encoder=False,
        eval_metric='logloss', scale_pos_weight=sum(y_bin==0)/max(sum(y_bin==1),1),
        random_state=42, verbosity=0,
    )
    preds_bin = cross_val_predict(xgb_bin, X_sc, y_bin, cv=cv, method='predict')
    f1 = f1_score(y_bin, preds_bin, average='binary', zero_division=0)
    proba_bin = cross_val_predict(xgb_bin, X_sc, y_bin, cv=cv, method='predict_proba')
    auc_val = roc_auc_score(y_bin, proba_bin[:, 1]) if len(np.unique(y_bin)) > 1 else 0.5
    print(f"  XGB {label:<15} binary F1={f1:.4f}  AUC={auc_val:.4f}")

# Fit full XGBoost for SHAP and feature importance
xgb_clf.fit(X_sc, y3_enc, sample_weight=sample_weights)

# ── SHAP Analysis ────────────────────────────────────────────────────────
print("\nComputing SHAP values...")
if HAS_SHAP:
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_sc)
    # shap_values: list of arrays [n_samples x n_features] per class
    # for 3-class XGBoost, shape = (3, n_samples, n_features)
    if isinstance(shap_values, list):
        shap_any = np.abs(np.stack(shap_values, axis=0)).mean(axis=0)  # mean over classes
    else:
        shap_any = np.abs(shap_values)

    mean_shap = shap_any.mean(axis=0)
    shap_df   = pd.DataFrame({'feature': SELECTED, 'mean_|shap|': mean_shap})
    shap_df   = shap_df.sort_values('mean_|shap|', ascending=False)
    print("\n  SHAP Feature Importance (mean |SHAP| across all classes):")
    for _, r in shap_df.iterrows():
        print(f"    {r['feature']:<28}  mean|shap|={r['mean_|shap|']:.4f}")
else:
    # Fall back to XGBoost built-in importance
    shap_df = pd.DataFrame({
        'feature': SELECTED,
        'mean_|shap|': xgb_clf.feature_importances_
    }).sort_values('mean_|shap|', ascending=False)
    print("  (Using XGB feature importances as SHAP fallback)")
    for _, r in shap_df.iterrows():
        print(f"    {r['feature']:<28}  importance={r['mean_|shap|']:.4f}")

# ── Figure 1: Confusion matrices ─────────────────────────────────────────
print("\nFigure 1: Confusion matrices...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 13: Meta-Classifier Confusion Matrices (5-Fold CV)",
             fontsize=14, fontweight='bold', y=0.98)

for ax, (preds, title) in zip(axes, [
        (lr_preds3, 'Logistic Regression'),
        (xgb_preds_labels, 'XGBoost'),
]):
    cm = confusion_matrix(y3, preds, labels=LABELS)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_yticklabels(LABELS, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            txt_color = 'white' if v > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(v), ha='center', va='center',
                    fontsize=13, fontweight='bold', color=txt_color)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's13_fig1_confusion_matrices.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13_fig1_confusion_matrices.png")

# ── Figure 2: ROC curves per class (XGBoost) ─────────────────────────────
print("Figure 2: ROC curves per class (XGBoost OvR)...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Step 13: XGBoost ROC Curves (One-vs-Rest, 5-Fold CV)",
             fontsize=14, fontweight='bold', y=0.98)

for ax, (class_idx, class_name) in enumerate([(0,'Inlier'),(1,'Type1'),(2,'Type2')]):
    ax = axes[class_idx]
    y_bin_true = (y3_enc == class_idx).astype(int)
    y_score    = xgb_proba3[:, class_idx]
    fpr, tpr, _ = roc_curve(y_bin_true, y_score)
    auc_val = sk_auc(fpr, tpr)
    ax.plot(fpr, tpr, color=list(TYPE_COLORS.values())[class_idx], lw=2.5,
            label=f'AUC = {auc_val:.4f}')
    ax.plot([0,1],[0,1],'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{class_name} vs Rest', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3); ax.set_facecolor('#f8f8f8')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's13_fig2_roc_per_class.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13_fig2_roc_per_class.png")

# ── Figure 3: SHAP / Feature importance ──────────────────────────────────
print("Figure 3: Feature importance (SHAP)...")
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Step 13: Feature Importance for Outlier Detection\n"
             "(Mean |SHAP| value — contribution to prediction)",
             fontsize=13, fontweight='bold', y=0.98)

sorted_shap = shap_df.sort_values('mean_|shap|', ascending=True)
colors = ['#c62828' if 'div' in f or 'slope' in f else
          '#e65100' if 'mono' in f or 'jump' in f else
          '#1565c0' for f in sorted_shap['feature']]
bars = ax.barh(sorted_shap['feature'], sorted_shap['mean_|shap|'],
               color=colors, alpha=0.8)
ax.set_xlabel('Mean |SHAP| value (feature contribution)', fontsize=11)
ax.set_title('')
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('#f8f8f8')
for bar, val in zip(bars, sorted_shap['mean_|shap|']):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#c62828', alpha=0.8, label='Consensus/trajectory features'),
    Patch(facecolor='#e65100', alpha=0.8, label='Monotonicity features'),
    Patch(facecolor='#1565c0', alpha=0.8, label='Training signal features'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(RES, 's13_fig3_shap_importance.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13_fig3_shap_importance.png")

# ── Figure 4: SHAP beeswarm (if shap available) ───────────────────────────
if HAS_SHAP:
    print("Figure 4: SHAP beeswarm plot...")
    # Use class 1 (Type1) and class 2 (Type2) SHAP values
    shap_vals_list = explainer.shap_values(X_sc)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Step 13: SHAP Beeswarm — Feature Impact on Type1 and Type2 Prediction",
                 fontsize=13, fontweight='bold', y=0.98)

    for ax_idx, (class_idx, class_name) in enumerate([(1,'Type1'),(2,'Type2')]):
        ax = axes[ax_idx]
        sv = shap_vals_list[class_idx]  # (n_samples, n_features)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:10]  # top 10
        feat_names_top = [SELECTED[i] for i in order]
        sv_top = sv[:, order]

        # manual dot plot
        for fi, (fname, col_idx) in enumerate(zip(feat_names_top, order)):
            feat_vals = X_sc[:, col_idx]
            shap_col  = sv[:, col_idx]
            scatter = ax.scatter(shap_col, [fi] * len(shap_col),
                                 c=feat_vals, cmap='coolwarm',
                                 alpha=0.4, s=15, vmin=-2, vmax=2)

        ax.set_yticks(range(len(feat_names_top)))
        ax.set_yticklabels(feat_names_top, fontsize=9)
        ax.axvline(0, color='black', lw=1, alpha=0.5)
        ax.set_xlabel('SHAP value (impact on prediction)', fontsize=10)
        ax.set_title(f'Top Features for {class_name}', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        ax.set_facecolor('#f8f8f8')

    plt.colorbar(scatter, ax=axes[1], label='Feature value (standardised)',
                 fraction=0.03)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(RES, 's13_fig4_shap_beeswarm.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: s13_fig4_shap_beeswarm.png")
else:
    print("  Skipped Figure 4 (shap not installed)")

# ── Figure 5: Predicted probability distribution by true type ────────────
print("Figure 5: Predicted probability distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Step 13: XGBoost Predicted Probability by True Class",
             fontsize=14, fontweight='bold', y=0.98)

for ax, (class_idx, class_name) in enumerate([(0,'Inlier'),(1,'Type1'),(2,'Type2')]):
    ax = axes[class_idx]
    probs = xgb_proba3[:, class_idx]
    for otype in ['Inlier', 'Type1', 'Type2']:
        mask = y3 == otype
        ax.hist(probs[mask], bins=20, alpha=0.6, color=TYPE_COLORS[otype],
                label=otype, density=True)
    ax.set_xlabel(f'P({class_name})', fontsize=11)
    ax.set_ylabel('Density')
    ax.set_title(f'Probability for {class_name}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3); ax.set_facecolor('#f8f8f8')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's13_fig5_prob_distributions.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s13_fig5_prob_distributions.png")

# ── Save results ───────────────────────────────────────────────────────────
shap_df.to_csv(os.path.join(RES, 'shap_importance.csv'), index=False)
df['xgb_pred_type'] = le.inverse_transform(xgb_preds_labels if isinstance(xgb_preds_labels[0], int)
                                            else [le.transform([p])[0] for p in xgb_preds_labels])
df['xgb_prob_type1'] = xgb_proba3[:, 1]
df['xgb_prob_type2'] = xgb_proba3[:, 2]
df.to_csv(os.path.join(RES, 'records_step13.csv'), index=False)
print(f"\n  Saved: shap_importance.csv, records_step13.csv")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 13 COMPLETE -- KEY TAKEAWAYS")
print("="*65)

lr_report  = classification_report(y3, lr_preds3, target_names=LABELS, output_dict=True)
xgb_report = classification_report(y3, xgb_preds_labels, target_names=LABELS, output_dict=True)

print("\n  Logistic Regression:")
for cls in LABELS:
    r = lr_report[cls]
    print(f"    {cls:<10} P={r['precision']:.3f} R={r['recall']:.3f} "
          f"F1={r['f1-score']:.3f} (n={int(r['support'])})")

print("\n  XGBoost:")
for cls in LABELS:
    r = xgb_report[cls]
    print(f"    {cls:<10} P={r['precision']:.3f} R={r['recall']:.3f} "
          f"F1={r['f1-score']:.3f} (n={int(r['support'])})")

print(f"\n  Top-3 features by SHAP importance:")
for _, r in shap_df.head(3).iterrows():
    print(f"    {r['feature']:<28}")

print("\n  ANALYSIS COMPLETE (all 13 steps)")
print("  Output files in: results/outlier_analysis/")
