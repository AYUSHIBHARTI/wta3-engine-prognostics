"""
Step 10: Consensus Analysis — Divergence from Group Median
===========================================================
Goal: At each prediction cycle, compare a single model's prediction to
the consensus (median of all 6 models). Large sustained divergence from
consensus is a signal that a model may be unreliable.

Key ideas:
  - consensus_pred(t) = median of the 6 models at cycle t
  - divergence(t)     = model_pred(t) - consensus_pred(t)
  - Features per (exp, model) run:
      mean_div        -- average signed divergence
      mean_abs_div    -- average absolute divergence from consensus
      max_abs_div     -- maximum absolute divergence
      div_std         -- standard deviation of divergence (volatile?)
      div_sign_cons   -- fraction of cycles where sign of divergence is consistent
                         (i.e. model is persistently above OR below)
      early_div / late_div -- mean |divergence| in first/last 30 cycles
      div_growth_ratio     -- late_div / early_div
  - Compare across Inlier / Type1 / Type2
  - ROC analysis: can consensus divergence features detect outliers?
  - Combine with step9 metric for a joint signal
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
from scipy import stats

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(BASE, 'results', 'outlier_analysis', 'step10')
os.makedirs(RES, exist_ok=True)

STEP1_CSV = os.path.join(BASE, 'results', 'outlier_analysis', 'step1',
                          'outlier_type_labels.csv')
STEP9_CSV = os.path.join(BASE, 'results', 'outlier_analysis', 'step9',
                          'records_step9.csv')
CHK1  = os.path.join(BASE, 'results', 'step1_checkpoint.json')
PREDS = os.path.join(BASE, 'results', 'all_predictions.json')

MODELS = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
TYPE_COLORS = {'Inlier': '#2196F3', 'Type1': '#c62828', 'Type2': '#e65100'}

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data...")
labels_df = pd.read_csv(STEP1_CSV)
step9_df  = pd.read_csv(STEP9_CSV)

with open(CHK1) as f: ck1 = json.load(f)
with open(PREDS) as f: all_preds_raw = json.load(f)

total_cycles = ck1['total_cycles']
calibration  = ck1['calibration']
pred_points  = ck1['prediction_points']

true_rul = np.array([max(0, total_cycles - calibration - i)
                     for i in range(pred_points)], dtype=float)

# build lookup: {(exp_id, model): predictions}
all_preds = {}
for item in all_preds_raw:
    eid = item['experiment']
    for mname, preds in item['test_predictions'].items():
        all_preds[(eid, mname)] = np.array(preds, dtype=float)

exp_ids = sorted(set(k for k, _ in all_preds.keys()))
n_exp   = len(exp_ids)

# ── compute consensus and divergence ──────────────────────────────────────
print("Computing consensus predictions and divergence features...")

records = []

for eid in exp_ids:
    # collect all model predictions for this experiment
    model_preds = {}
    for mname in MODELS:
        p = all_preds.get((eid, mname))
        if p is not None and len(p) == pred_points:
            model_preds[mname] = p

    if len(model_preds) < 2:
        continue

    # consensus = median across models at each cycle
    pred_stack   = np.stack(list(model_preds.values()), axis=0)  # (n_models, n_cycles)
    consensus    = np.median(pred_stack, axis=0)                  # (n_cycles,)

    for mname, preds in model_preds.items():
        divergence = preds - consensus            # signed
        abs_div    = np.abs(divergence)

        mean_div       = float(np.mean(divergence))
        mean_abs_div   = float(np.mean(abs_div))
        max_abs_div    = float(np.max(abs_div))
        div_std        = float(np.std(divergence))

        # sign consistency: fraction of cycles with positive divergence
        frac_pos = float((divergence > 0).mean())
        # consistent if mostly above (>0.7) or mostly below (<0.3)
        div_sign_cons  = float(max(frac_pos, 1 - frac_pos))

        # early / late divergence
        n30 = min(30, pred_points // 4)
        early_div = float(np.mean(abs_div[:n30]))
        late_div  = float(np.mean(abs_div[-n30:]))
        div_growth_ratio = late_div / max(early_div, 1e-6)

        # mean divergence in each RUL zone
        zones = {
            'very_early': true_rul > 150,
            'early':      (true_rul > 100) & (true_rul <= 150),
            'mid':        (true_rul > 50)  & (true_rul <= 100),
            'late':       (true_rul > 20)  & (true_rul <= 50),
            'critical':   true_rul <= 20,
        }
        zone_divs = {}
        for zname, zmask in zones.items():
            if zmask.sum() > 0:
                zone_divs[f'abs_div_{zname}'] = float(np.mean(abs_div[zmask]))
            else:
                zone_divs[f'abs_div_{zname}'] = np.nan

        row   = labels_df[(labels_df['experiment'] == eid) &
                          (labels_df['model'] == mname)]
        otype = row['otype'].values[0] if len(row) else 'Inlier'

        rec = dict(
            experiment=eid, model=mname, outlier_type=otype,
            mean_div=mean_div,
            mean_abs_div=mean_abs_div,
            max_abs_div=max_abs_div,
            div_std=div_std,
            div_sign_cons=div_sign_cons,
            early_div=early_div,
            late_div=late_div,
            div_growth_ratio=div_growth_ratio,
        )
        rec.update(zone_divs)
        records.append(rec)

df = pd.DataFrame(records)
print(f"  Records: {len(df)}")

# ── summary statistics ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  CONSENSUS DIVERGENCE SUMMARY BY OUTLIER TYPE")
print("="*70)
print(f"\n{'Type':<10} {'N':>5} {'mean_abs_div':>14} {'max_abs_div':>14} {'div_growth':>12}")
print("-"*60)
for otype in ['Inlier', 'Type1', 'Type2']:
    sub = df[df['outlier_type'] == otype]
    if len(sub) == 0:
        continue
    print(f"{otype:<10} {len(sub):>5} "
          f"{sub['mean_abs_div'].mean():>14.4f} "
          f"{sub['max_abs_div'].mean():>14.4f} "
          f"{sub['div_growth_ratio'].mean():>12.4f}")

# ── ROC analysis ───────────────────────────────────────────────────────────
print("\n  ROC-AUC for consensus divergence features:")

def compute_roc_auc(scores, y_true):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return sk_auc(fpr, tpr), fpr, tpr

feat_map = {
    'mean_abs_div':    df['mean_abs_div'],
    'max_abs_div':     df['max_abs_div'],
    'div_growth_ratio':df['div_growth_ratio'],
    'div_sign_cons':   df['div_sign_cons'],
    'late_div':        df['late_div'],
}

roc_results = {}
for fname, scores in feat_map.items():
    y1 = (df['outlier_type'] == 'Type1').astype(int)
    y2 = (df['outlier_type'] == 'Type2').astype(int)
    a1, fpr1, tpr1 = compute_roc_auc(scores.fillna(0), y1)
    a2, fpr2, tpr2 = compute_roc_auc(scores.fillna(0), y2)
    roc_results[fname] = dict(auc1=a1, fpr1=fpr1, tpr1=tpr1,
                               auc2=a2, fpr2=fpr2, tpr2=tpr2)
    print(f"  {fname:<22}  Type1 AUC={a1:.4f}  Type2 AUC={a2:.4f}")

# ── Combined with Step 9 inter-model correlation ───────────────────────────
print("\n  Combined signal (divergence + neg inter-model corr):")
merged = df.merge(step9_df[['experiment','model','mean_corr_with_others']],
                  on=['experiment','model'], how='left')
merged['combined_score'] = (merged['mean_abs_div'] - merged['mean_corr_with_others'])

for otype_key, label in [('Type1', 'Type1'), ('Type2', 'Type2')]:
    y = (merged['outlier_type'] == otype_key).astype(int)
    a, _, _ = compute_roc_auc(merged['combined_score'].fillna(0), y)
    print(f"  combined (abs_div - mean_corr)  {label} AUC={a:.4f}")

# ── Figure 1: boxplots of divergence features ─────────────────────────────
print("\nFigure 1: Divergence feature distributions...")
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Step 10: Consensus Divergence Features by Outlier Type",
             fontsize=14, fontweight='bold', y=0.98)

for ax, (feat, label) in zip(axes, [
        ('mean_abs_div',     'Mean |Divergence| from Consensus'),
        ('max_abs_div',      'Max |Divergence| from Consensus'),
        ('div_growth_ratio', 'Divergence Growth Ratio\n(late/early)'),
        ('div_sign_cons',    'Sign Consistency\n(1=always same side)'),
]):
    data = [df[df['outlier_type'] == t][feat].dropna().values
            for t in ['Inlier', 'Type1', 'Type2']]
    bp = ax.boxplot(data, tick_labels=['Inlier', 'Type1', 'Type2'],
                    patch_artist=True, notch=False, showfliers=True)
    for patch, color in zip(bp['boxes'],
                             [TYPE_COLORS[t] for t in ['Inlier','Type1','Type2']]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f8f8f8')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's10_fig1_div_distributions.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s10_fig1_div_distributions.png")

# ── Figure 2: divergence by RUL zone ─────────────────────────────────────
print("Figure 2: Divergence by RUL zone...")
zones_order = ['very_early', 'early', 'mid', 'late', 'critical']
zone_labels = ['>150', '100-150', '50-100', '20-50', '0-20']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Step 10: Mean |Divergence| from Consensus by RUL Zone",
             fontsize=14, fontweight='bold', y=0.98)

for ax, otype in zip(axes, ['Inlier', 'Type1', 'Type2']):
    sub = df[df['outlier_type'] == otype]
    means = [sub[f'abs_div_{z}'].mean() for z in zones_order]
    stds  = [sub[f'abs_div_{z}'].std()  for z in zones_order]
    bars  = ax.bar(zone_labels, means, color=TYPE_COLORS[otype], alpha=0.7,
                   yerr=stds, capsize=5)
    ax.set_title(f'{otype} (n={len(sub)})', fontsize=11)
    ax.set_xlabel('RUL Zone')
    ax.set_ylabel('Mean |Divergence| (cycles)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's10_fig2_div_by_zone.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s10_fig2_div_by_zone.png")

# ── Figure 3: ROC curves ──────────────────────────────────────────────────
print("Figure 3: ROC curves...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 10: ROC Curves — Consensus Divergence Features",
             fontsize=14, fontweight='bold', y=0.98)

feat_styles = {
    'mean_abs_div':     ('-',  '#c62828', 'Mean |Div|'),
    'max_abs_div':      ('--', '#e65100', 'Max |Div|'),
    'div_growth_ratio': ('-.',  '#6a1b9a', 'Growth Ratio'),
    'div_sign_cons':    (':',  '#1565c0', 'Sign Consistency'),
    'late_div':         ('-',  '#2e7d32', 'Late |Div|'),
}

for ax, (type_key, title) in zip(axes, [('1', 'Type1 Detection'), ('2', 'Type2 Detection')]):
    ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random (0.50)')
    for fname, (ls, color, label) in feat_styles.items():
        if fname not in roc_results:
            continue
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
plt.savefig(os.path.join(RES, 's10_fig3_roc_curves.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s10_fig3_roc_curves.png")

# ── Figure 4: consensus trajectory examples (Type1 vs Inlier) ────────────
print("Figure 4: Example divergence trajectories...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Step 10: Example Consensus Divergence Trajectories",
             fontsize=14, fontweight='bold', y=0.98)

cycles = np.arange(calibration + 1, calibration + 1 + pred_points)

# pick one Type1 experiment
type1_rows = df[df['outlier_type'] == 'Type1']
type1_inlier = df[df['outlier_type'] == 'Inlier']

for ax, otype, rows, title in zip(axes,
        ['Type1', 'Inlier'],
        [type1_rows, type1_inlier],
        ['Type1 (XGB) — Divergence from Consensus',
         'Inlier (RF) — Divergence from Consensus']):
    if len(rows) == 0:
        ax.set_visible(False)
        continue

    sample_rows = rows.head(min(5, len(rows)))
    for _, r in sample_rows.iterrows():
        eid   = r['experiment']
        mname = r['model']
        preds = all_preds.get((eid, mname))
        if preds is None:
            continue
        # get consensus for this experiment
        mp = {}
        for mn in MODELS:
            p = all_preds.get((eid, mn))
            if p is not None and len(p) == pred_points:
                mp[mn] = p
        if len(mp) < 2:
            continue
        cons = np.median(np.stack(list(mp.values()), axis=0), axis=0)
        div  = preds - cons
        ax.plot(cycles, div, alpha=0.7, lw=1.5,
                label=f'{eid} {mname}')

    ax.axhline(0, color='black', lw=1.5, linestyle='--', alpha=0.6,
               label='Consensus (0)')
    ax.fill_between(cycles, -5, 5, alpha=0.1, color='green',
                    label='+-5 cycle band')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Divergence from Consensus (cycles)')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f8f8')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RES, 's10_fig4_trajectory_examples.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s10_fig4_trajectory_examples.png")

# ── Figure 5: 2D scatter — mean_abs_div vs mean_corr_with_others ──────────
print("Figure 5: 2D scatter — divergence vs inter-model correlation...")
fig, ax = plt.subplots(figsize=(10, 6))
for otype in ['Inlier', 'Type2', 'Type1']:
    sub = merged[merged['outlier_type'] == otype]
    ax.scatter(sub['mean_corr_with_others'], sub['mean_abs_div'],
               c=TYPE_COLORS[otype], label=otype,
               alpha=0.5, s=40 if otype == 'Inlier' else 80,
               edgecolors='none')
ax.set_xlabel('Mean Correlation with Other Models (Step 9)', fontsize=12)
ax.set_ylabel('Mean |Divergence| from Consensus (Step 10)', fontsize=12)
ax.set_title('Step 10: Joint Signal — Outliers cluster\nhigh-divergence AND low-correlation',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_facecolor('#f8f8f8')
plt.tight_layout()
plt.savefig(os.path.join(RES, 's10_fig5_2d_signal.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: s10_fig5_2d_signal.png")

# ── Save CSV ───────────────────────────────────────────────────────────────
out_csv = os.path.join(RES, 'records_step10.csv')
df.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv}")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 10 COMPLETE -- KEY TAKEAWAYS")
print("="*65)

for otype in ['Inlier', 'Type1', 'Type2']:
    sub = df[df['outlier_type'] == otype]
    if len(sub) == 0:
        continue
    print(f"\n  {otype} ({len(sub)} runs):")
    print(f"    mean_abs_div   = {sub['mean_abs_div'].mean():.4f} cycles")
    print(f"    max_abs_div    = {sub['max_abs_div'].mean():.4f} cycles")
    print(f"    div_growth     = {sub['div_growth_ratio'].mean():.4f}x")
    print(f"    sign_cons      = {sub['div_sign_cons'].mean():.4f}")

best_f1 = max(roc_results, key=lambda x: roc_results[x]['auc1'])
best_f2 = max(roc_results, key=lambda x: roc_results[x]['auc2'])
print(f"\n  Best detector for Type1: {best_f1} (AUC={roc_results[best_f1]['auc1']:.4f})")
print(f"  Best detector for Type2: {best_f2} (AUC={roc_results[best_f2]['auc2']:.4f})")
print(f"\n  -> Next: Step 11 -- Feature importance ranking")
print(f"  -> Output files in: {RES}/")
