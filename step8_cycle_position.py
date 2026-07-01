# =============================================================================
# step8_cycle_position.py
# Analyse how prediction error relates to cycle position (where in the engine
# lifecycle we are). Cycle position is always available at inference time —
# no model, no ensemble, no OOD computation needed.
# Key question: is knowing "how far along the engine is" enough to flag risk?
# =============================================================================

import json
import os
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr, spearmanr

plt.style.use('default')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

RESULTS_DIR      = 'results'
STEP4_DIR        = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step4')
STEP7_DIR        = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step7')
OUT_DIR          = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step8')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES      = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
COLORS           = {'Type1': '#D32F2F', 'Type2': '#1976D2', 'Inlier': '#388E3C'}
PRED_START_CYCLE = 61
TOTAL_CYCLES     = 213
N_PRED           = TOTAL_CYCLES - PRED_START_CYCLE + 1

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
with open(os.path.join(RESULTS_DIR, 'all_predictions.json')) as f:
    predictions = json.load(f)

records = []
with open(os.path.join(STEP4_DIR, 'records_with_trajectory.csv')) as f:
    for row in csv.DictReader(f):
        for col in ['val_rmse','train_rmse','gap_ratio','val_rmse_zscore',
                    'gap_zscore','test_mae','mono_violations','mono_rate',
                    'max_jump','mean_abs_diff','mean_rolling_var','mean_ma_dev',
                    'slope','slope_dev','slope_consistency','final_pred']:
            try:
                row[col] = float(row[col])
            except (ValueError, KeyError):
                row[col] = np.nan
        row['outlier'] = row['outlier'] == 'True'
        records.append(row)

dist_df  = pd.read_csv(os.path.join(STEP7_DIR, 'mahalanobis_distances.csv'))
mah_dist = dist_df['mahalanobis_dist'].values

y_true   = np.array(predictions[0]['y_test'])
cycles   = np.arange(PRED_START_CYCLE, TOTAL_CYCLES + 1)

# Cycle position features
cycle_norm   = (cycles - PRED_START_CYCLE) / (TOTAL_CYCLES - PRED_START_CYCLE)
rul_norm     = y_true / y_true.max()     # normalised RUL (1=start, 0=end)
rul_inv      = 1.0 - rul_norm            # degradation progress (0=start, 1=end)

# Error store
error_store = {}
for pred_row in predictions:
    exp = pred_row['experiment']
    for m in MODEL_NAMES:
        error_store[(exp, m)] = np.abs(
            np.array(pred_row['test_predictions'][m]) - y_true)

signed_store = {}
for pred_row in predictions:
    exp = pred_row['experiment']
    for m in MODEL_NAMES:
        signed_store[(exp, m)] = (
            np.array(pred_row['test_predictions'][m]) - y_true)

def get_errors(otype, model=None, signed=False):
    store = signed_store if signed else error_store
    recs  = [r for r in records if r['otype'] == otype]
    if model:
        recs = [r for r in recs if r['model'] == model]
    arrays = [store[(r['experiment'], r['model'])]
              for r in recs if (r['experiment'], r['model']) in store]
    return np.array(arrays) if arrays else np.empty((0, N_PRED))

# ═════════════════════════════════════════════════════════════════════════════
# PRINT: correlations between cycle features and error
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  CORRELATION: CYCLE POSITION vs MEAN |ERROR|")
print("=" * 70)
print(f"\n{'Feature':<18} {'Type':<10} {'Pearson':>9} {'Spearman':>10} {'Interpretation'}")
print("-" * 65)

for otype in ['Inlier', 'Type1', 'Type2']:
    mat = get_errors(otype)
    if mat.size == 0:
        continue
    mean_err = mat.mean(axis=0)
    for fname, fvals in [('cycle_norm', cycle_norm),
                          ('rul_inverse', rul_inv),
                          ('true_rul', y_true),
                          ('mahal_dist', mah_dist)]:
        pr, _  = pearsonr(fvals, mean_err)
        sr, _  = spearmanr(fvals, mean_err)
        interp = ('strong' if abs(pr) > 0.6 else
                  'moderate' if abs(pr) > 0.35 else 'weak')
        print(f"  {fname:<16} {otype:<10} {pr:>9.4f} {sr:>10.4f}  {interp}")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Error vs cycle position: scatter with regression per type
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 1: Error vs cycle position...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, otype, title in [
    (axes[0], 'Inlier', 'Inlier'),
    (axes[1], 'Type1',  'Type 1 (Bad XGB)'),
    (axes[2], 'Type2',  'Type 2 (Divergent)')]:

    mat = get_errors(otype)
    if mat.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        continue
    mean_err = mat.mean(axis=0)
    pr, _    = pearsonr(cycle_norm, mean_err)

    sc = ax.scatter(cycle_norm, mean_err, c=y_true, cmap='RdYlGn',
                    s=25, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='True RUL')

    coef  = np.polyfit(cycle_norm, mean_err, 2)   # quadratic fit
    xline = np.linspace(0, 1, 100)
    ax.plot(xline, np.polyval(coef, xline), 'k-', linewidth=2,
            label=f'Quadratic fit  r={pr:.3f}')
    ax.set_xlabel('Normalised Cycle Position  (0=start, 1=end of life)',
                  fontsize=9)
    ax.set_ylabel('Mean |Error| (cycles)', fontsize=9)
    ax.set_title(f'{title} (n={len(mat)} runs)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

fig.suptitle('Step 8 — Error vs Cycle Position\n'
             '(does prediction error grow as the engine approaches end of life?)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's8_fig1_error_vs_cycle_position.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s8_fig1_error_vs_cycle_position.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Error by decile of cycle position
#            Shows exactly where in the life the error spikes
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 2: Error by cycle position decile...")
decile_edges  = np.percentile(cycle_norm, np.arange(0, 110, 10))
decile_labels = [f'{int(i*10)}-{int((i+1)*10)}%' for i in range(10)]
decile_masks  = [(cycle_norm >= decile_edges[i]) & (cycle_norm < decile_edges[i+1])
                 for i in range(10)]
decile_masks[-1] = (cycle_norm >= decile_edges[-2])   # include last point

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, ylabel, signed in [
    (axes[0], 'abs',    'Mean |Error| (cycles)',      False),
    (axes[1], 'signed', 'Mean Signed Error (cycles)', True)]:

    x = np.arange(10)
    w = 0.25
    for i_ot, (otype, color) in enumerate([('Inlier','#388E3C'),
                                             ('Type1','#D32F2F'),
                                             ('Type2','#1976D2')]):
        mat  = get_errors(otype, signed=signed)
        if mat.size == 0:
            continue
        mean_err  = mat.mean(axis=0)
        dec_means = [mean_err[dm].mean() for dm in decile_masks]
        ax.bar(x + (i_ot-1)*w, dec_means, w, color=color,
               alpha=0.85, label=otype, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(decile_labels, fontsize=8, rotation=30)
    ax.set_xlabel('Cycle Position Decile', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if signed:
        ax.axhline(0, color='black', linewidth=1)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.set_title('Absolute Error by Decile' if not signed
                 else 'Signed Error by Decile', fontsize=11)

fig.suptitle('Step 8 — Error Breakdown by Cycle Position Decile\n'
             '(0% = test window start (cycle 61), 100% = end of life)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's8_fig2_error_by_decile.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s8_fig2_error_by_decile.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ROC: can cycle position alone predict high-error cycles?
#            Compare: cycle_norm vs rul_inverse vs mahalanobis
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 3: ROC comparison of cycle features...")
ERROR_THRESHOLD = 10.0
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, otype, title in [
    (axes[0], 'Inlier', 'Inlier'),
    (axes[1], 'Type1',  'Type 1'),
    (axes[2], 'Type2',  'Type 2')]:

    mat       = get_errors(otype)
    if mat.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        continue

    mean_err  = mat.mean(axis=0)
    y_high    = (mean_err > ERROR_THRESHOLD).astype(int)

    if y_high.sum() == 0 or y_high.sum() == len(y_high):
        ax.text(0.5, 0.5, 'All same class', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        continue

    for fname, fvals, color, ls in [
        ('cycle_norm',   cycle_norm, '#1565C0', '-'),
        ('rul_inverse',  rul_inv,    '#E65100', '--'),
        ('mahal_dist',   mah_dist,   '#6A1B9A', '-.'),
        ('combined',     cycle_norm * mah_dist, '#2E7D32', ':')]:

        fpr, tpr, _ = roc_curve(y_high, fvals)
        roc_auc     = auc(fpr, tpr)
        if roc_auc < 0.5:
            fpr, tpr, _ = roc_curve(y_high, -fvals)
            roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, ls, linewidth=2, color=color,
                label=f'{fname}  AUC={roc_auc:.3f}')

    ax.plot([0,1],[0,1],'k--', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.set_title(f'{title}\n(high error > {ERROR_THRESHOLD} cycles)', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(alpha=0.25)

fig.suptitle('Step 8 — ROC: Cycle Position vs Mahalanobis vs Combined\n'
             'as Predictors of High-Error Cycles',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's8_fig3_roc_cycle_vs_mahal.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s8_fig3_roc_cycle_vs_mahal.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-model: correlation of cycle_norm and error
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 4: Per-model correlation matrix...")
fig, ax = plt.subplots(figsize=(11, 5))

features    = {'cycle_norm': cycle_norm, 'rul_inverse': rul_inv,
               'mahal_dist': mah_dist,   'combined': cycle_norm * mah_dist}
otype_list  = ['Inlier', 'Type1', 'Type2']

# Build (model x feature) correlation matrix for each otype
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, otype in zip(axes, otype_list):
    mat_vals = np.full((len(MODEL_NAMES), len(features)), np.nan)
    for mi, m in enumerate(MODEL_NAMES):
        errs = get_errors(otype, model=m)
        if errs.size == 0:
            continue
        mean_err = errs.mean(axis=0)
        for fi, (fname, fvals) in enumerate(features.items()):
            pr, _ = pearsonr(fvals, mean_err)
            mat_vals[mi, fi] = pr

    im = ax.imshow(mat_vals, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(list(features.keys()), fontsize=8, rotation=20)
    ax.set_yticklabels(MODEL_NAMES, fontsize=9)
    ax.set_title(otype, fontsize=11)
    for mi in range(len(MODEL_NAMES)):
        for fi in range(len(features)):
            v = mat_vals[mi, fi]
            if not np.isnan(v):
                ax.text(fi, mi, f'{v:.2f}', ha='center', va='center',
                        fontsize=8,
                        color='white' if abs(v) > 0.6 else 'black')
            else:
                ax.text(fi, mi, 'N/A', ha='center', va='center',
                        fontsize=7, color='gray')

fig.suptitle('Step 8 — Correlation: Cycle Features vs Error  (per model per type)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's8_fig4_per_model_correlation.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s8_fig4_per_model_correlation.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Risk zones: combine cycle position + Mahalanobis into a 2D map
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5: 2D risk map (cycle position x Mahalanobis)...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, otype, title in [
    (axes[0], 'Inlier', 'Inlier'),
    (axes[1], 'Type1',  'Type 1 (Bad)'),
    (axes[2], 'Type2',  'Type 2 (Divergent)')]:

    mat      = get_errors(otype)
    if mat.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        continue
    mean_err = mat.mean(axis=0)

    sc = ax.scatter(cycle_norm, mah_dist, c=mean_err,
                    cmap='RdYlGn_r', s=35, alpha=0.9,
                    vmin=0, vmax=mean_err.max())
    plt.colorbar(sc, ax=ax, label='Mean |Error| (cycles)')
    ax.set_xlabel('Cycle Position (0=start, 1=end)', fontsize=9)
    ax.set_ylabel('Mahalanobis Distance', fontsize=9)
    ax.set_title(f'{title}', fontsize=10)
    ax.grid(alpha=0.2)

fig.suptitle('Step 8 — 2D Risk Map: Cycle Position × Mahalanobis Distance\n'
             '(red = high error — do risky cycles cluster in specific regions?)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's8_fig5_2d_risk_map.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s8_fig5_2d_risk_map.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Summary: which cycle-level features add unique info?
#            Partial correlations controlling for true RUL
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 6: Partial correlations controlling for RUL...")
from numpy.linalg import lstsq

def partial_corr(x, y, z):
    """Correlation between x and y after removing linear effect of z."""
    def residual(a, b):
        b_col    = b.reshape(-1, 1)
        coef, _, _, _ = lstsq(
            np.column_stack([b_col, np.ones(len(b))]), a, rcond=None)
        return a - np.column_stack([b_col, np.ones(len(b))]) @ coef
    rx = residual(x, z)
    ry = residual(y, z)
    return pearsonr(rx, ry)[0]

fig, ax = plt.subplots(figsize=(11, 5))
feat_names = ['cycle_norm', 'mahal_dist', 'combined']
feat_vals  = [cycle_norm,   mah_dist,     cycle_norm * mah_dist]

x_pos  = np.arange(len(feat_names))
w      = 0.25
bar_data = {ot: [] for ot in ['Inlier','Type1','Type2']}

for otype in ['Inlier', 'Type1', 'Type2']:
    mat = get_errors(otype)
    if mat.size == 0:
        bar_data[otype] = [0] * len(feat_names)
        continue
    mean_err = mat.mean(axis=0)
    for fvals in feat_vals:
        pc = partial_corr(fvals, mean_err, y_true.astype(float))
        bar_data[otype].append(pc)

for i_ot, (otype, color) in enumerate([('Inlier','#388E3C'),
                                         ('Type1','#D32F2F'),
                                         ('Type2','#1976D2')]):
    ax.bar(x_pos + (i_ot-1)*w, bar_data[otype], w,
           color=color, alpha=0.85, label=otype,
           edgecolor='white', linewidth=0.4)

ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(feat_names, fontsize=10)
ax.set_ylabel('Partial Correlation (controlling for true RUL)', fontsize=10)
ax.set_title('Step 8 — Partial Correlations: Unique Info Beyond "How Close to End of Life?"\n'
             '(if near-zero: feature only tells us about lifecycle phase, not reliability)',
             fontsize=11, pad=10)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's8_fig6_partial_correlations.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s8_fig6_partial_correlations.png")

# ── Save enriched records ─────────────────────────────────────────────────────
# Add per-run cycle-level summary features
print("\nComputing per-run cycle summary features...")
for r in records:
    key = (r['experiment'], r['model'])
    if key not in error_store:
        r['mean_err_early']  = np.nan
        r['mean_err_late']   = np.nan
        r['err_growth_ratio']= np.nan
        r['corr_err_cycle']  = np.nan
        r['corr_err_mahal']  = np.nan
        continue
    errs     = error_store[key]
    early_30 = errs[:30].mean()
    late_30  = errs[-30:].mean()
    r['mean_err_early']   = float(early_30)
    r['mean_err_late']    = float(late_30)
    r['err_growth_ratio'] = float(late_30 / early_30) if early_30 > 0 else 1.0
    r['corr_err_cycle']   = float(pearsonr(cycle_norm, errs)[0])
    r['corr_err_mahal']   = float(pearsonr(mah_dist,   errs)[0])

csv_path = os.path.join(OUT_DIR, 'records_step8.csv')
all_keys = list(records[0].keys())
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore', restval='')
    writer.writeheader()
    writer.writerows(records)
print(f"  Saved: {csv_path}")

# ── Final summary ──────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  STEP 8 COMPLETE -- KEY TAKEAWAYS")
print("=" * 65)
for otype in ['Inlier', 'Type1', 'Type2']:
    mat = get_errors(otype)
    if mat.size == 0:
        continue
    mean_err = mat.mean(axis=0)
    pr_cyc,  _ = pearsonr(cycle_norm, mean_err)
    pr_mah,  _ = pearsonr(mah_dist,   mean_err)
    pr_comb, _ = pearsonr(cycle_norm * mah_dist, mean_err)
    print(f"\n  {otype} ({len(mat)} runs):")
    print(f"    corr(cycle_norm, error)    = {pr_cyc:+.4f}")
    print(f"    corr(mahal_dist, error)    = {pr_mah:+.4f}")
    print(f"    corr(cycle x mahal, error) = {pr_comb:+.4f}")
    print(f"    Early (first 30) error     = {mean_err[:30].mean():.2f} cycles")
    print(f"    Late  (last  30) error     = {mean_err[-30:].mean():.2f} cycles")
    print(f"    Growth ratio (late/early)  = "
          f"{mean_err[-30:].mean()/mean_err[:30].mean():.2f}x")

print()
print("  -> Next: Step 9 -- Inter-model error correlation matrix")
print(f"  -> Output files in: {OUT_DIR}/")
