# =============================================================================
# step7_mahalanobis.py
# Compute Mahalanobis distance of each test cycle's sensor readings from
# the training distribution. Answers: does input OOD-ness correlate with
# prediction error and outlier behaviour?
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
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
from sklearn.metrics import roc_curve, auc

plt.style.use('default')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

RESULTS_DIR      = 'results'
STEP4_DIR        = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step4')
OUT_DIR          = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step7')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES      = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
COLORS           = {'Type1': '#D32F2F', 'Type2': '#1976D2', 'Inlier': '#388E3C'}
DATA_PATH        = 'data/train_FD001.txt'
PRED_START_CYCLE = 61
TOTAL_CYCLES     = 213
N_PRED           = TOTAL_CYCLES - PRED_START_CYCLE + 1
MASTER_SEED      = 42
ROLLING_WINDOW   = 50
TOP_SENSORS      = ['sensor_11', 'sensor_12', 'sensor_4', 'sensor_7', 'sensor_15']
TEST_ENGINE      = 52

# ── Load records ──────────────────────────────────────────────────────────────
print("Loading records and predictions...")
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

y_true = np.array(predictions[0]['y_test'])
cycles = np.arange(PRED_START_CYCLE, TOTAL_CYCLES + 1)

error_store = {}
for pred_row in predictions:
    exp = pred_row['experiment']
    for m in MODEL_NAMES:
        error_store[(exp, m)] = np.abs(
            np.array(pred_row['test_predictions'][m]) - y_true)

# ── Reconstruct features ──────────────────────────────────────────────────────
print("Reconstructing features from raw data...")
col_names = (['engine_id', 'cycle'] +
             [f'setting_{i}' for i in range(1, 4)] +
             [f'sensor_{i}'  for i in range(1, 22)])
df = pd.read_csv(DATA_PATH, sep=r'\s+', header=None, names=col_names)
max_cyc = df.groupby('engine_id')['cycle'].max().reset_index()
max_cyc.columns = ['engine_id', 'max_cycle']
df = df.merge(max_cyc, on='engine_id')
df['RUL'] = df['max_cycle'] - df['cycle']
df = df.drop('max_cycle', axis=1)

# Identical train/val split from step5_evaluation.py
np.random.seed(MASTER_SEED)
legacy         = df[df['engine_id'] != TEST_ENGINE].copy()
engine_max_cyc = legacy.groupby('engine_id')['cycle'].max()
quartiles      = pd.qcut(engine_max_cyc, q=4, labels=['Q1','Q2','Q3','Q4'])
train_engines, val_engines = [], []
for q in ['Q1','Q2','Q3','Q4']:
    q_ids = quartiles[quartiles == q].index.tolist()
    np.random.shuffle(q_ids)
    val_engines.extend(q_ids[:3])
    train_engines.extend(q_ids[3:])

train_data  = legacy[legacy['engine_id'].isin(train_engines)].copy()
engine_data = df[df['engine_id'] == TEST_ENGINE].copy()

def add_features(d):
    d = d.copy().sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    d['cycle_norm']    = d.groupby('engine_id')['cycle'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
    d['cycle_squared'] = d['cycle_norm'] ** 2
    for s in TOP_SENSORS:
        d[f'{s}_roll_mean'] = d.groupby('engine_id')[s].transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
        d[f'{s}_roll_std']  = d.groupby('engine_id')[s].transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))
        d[f'{s}_roll_diff'] = d.groupby('engine_id')[s].transform(
            lambda x: x.diff().rolling(ROLLING_WINDOW, min_periods=1).mean().fillna(0))
    return d

rolling_cols = [f'{s}{suf}' for s in TOP_SENSORS
                for suf in ('_roll_mean','_roll_std','_roll_diff')]
feature_cols = TOP_SENSORS + ['cycle_norm','cycle_squared'] + rolling_cols

train_data  = add_features(train_data)
engine_data = add_features(engine_data)

scaler = RobustScaler()
X_train = scaler.fit_transform(train_data[feature_cols].values.astype(np.float32))
engine_data[feature_cols] = scaler.transform(
    engine_data[feature_cols].values.astype(np.float32))

# Test window: cycles 61–213
test_data = engine_data[engine_data['cycle'] >= PRED_START_CYCLE].copy()
X_test    = test_data[feature_cols].values.astype(np.float32)
assert len(X_test) == N_PRED, f"Expected {N_PRED} test cycles, got {len(X_test)}"
print(f"  Train: {X_train.shape}   Test: {X_test.shape}")

# ── Compute Mahalanobis distance ──────────────────────────────────────────────
print("Computing Mahalanobis distance for each test cycle...")
# Use PCA to reduce dimensionality before inversion (avoids singular covariance)
pca       = PCA(n_components=10, random_state=42)
X_tr_pca  = pca.fit_transform(X_train)
X_te_pca  = pca.transform(X_test)

mu_train  = X_tr_pca.mean(axis=0)
cov_train = np.cov(X_tr_pca.T)
VI        = pinv(cov_train)   # pseudoinverse — robust to near-singular matrices

mah_dist  = np.array([mahalanobis(x, mu_train, VI) for x in X_te_pca])
print(f"  Mahalanobis distance range: [{mah_dist.min():.2f}, {mah_dist.max():.2f}]")
print(f"  Mean={mah_dist.mean():.2f}  Std={mah_dist.std():.2f}\n")

# ── Print correlation: distance vs true RUL ───────────────────────────────────
corr_rul  = np.corrcoef(mah_dist, y_true)[0, 1]
print(f"  Correlation (Mahalanobis, true RUL) = {corr_rul:.4f}")
print(f"  -> {'Strong' if abs(corr_rul)>0.7 else 'Moderate' if abs(corr_rul)>0.4 else 'Weak'} "
      f"relationship: distance {'increases' if corr_rul>0 else 'decreases'} as RUL increases\n")

# ── Cycle-level error vs Mahalanobis ─────────────────────────────────────────
print("=" * 65)
print("  CORRELATION: MAHALANOBIS DISTANCE vs PREDICTION ERROR")
print("=" * 65)
print(f"\n{'Model':<14} {'Type':<10} {'Corr(dist, |error|)':>20}")
print("-" * 48)
for m in MODEL_NAMES:
    for otype in ['Inlier', 'Type1', 'Type2']:
        keys = [(r['experiment'], m) for r in records
                if r['model'] == m and r['otype'] == otype]
        if not keys:
            continue
        errs = np.array([error_store[k] for k in keys if k in error_store])
        if errs.size == 0:
            continue
        mean_err = errs.mean(axis=0)
        corr = np.corrcoef(mah_dist, mean_err)[0, 1]
        flag = ' <-- notable' if abs(corr) > 0.5 else ''
        print(f"  {m:<12} {otype:<10} {corr:>18.4f}{flag}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Mahalanobis distance vs cycle and RUL
# ═════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Mahalanobis distance profile...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(cycles, mah_dist, color='#1565C0', linewidth=1.5, label='Mahalanobis dist')
ax.fill_between(cycles, 0, mah_dist, alpha=0.15, color='#1565C0')
ax.axhline(mah_dist.mean(), color='gray', linestyle='--', linewidth=1,
           label=f'Mean = {mah_dist.mean():.2f}')
ax.axhline(mah_dist.mean() + 2*mah_dist.std(), color='orange', linestyle=':',
           linewidth=1, label=f'Mean+2σ = {mah_dist.mean()+2*mah_dist.std():.2f}')
ax.set_xlabel('Cycle', fontsize=10)
ax.set_ylabel('Mahalanobis Distance from Training Centroid', fontsize=10)
ax.set_title('Distance Profile Across Test Cycles', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

ax = axes[1]
sc = ax.scatter(y_true, mah_dist, c=cycles, cmap='viridis', s=30, alpha=0.8)
plt.colorbar(sc, ax=ax, label='Cycle number')
ax.set_xlabel('True RUL (cycles remaining)', fontsize=10)
ax.set_ylabel('Mahalanobis Distance', fontsize=10)
ax.set_title(f'Distance vs True RUL\n(corr = {corr_rul:.3f})', fontsize=11)
ax.grid(alpha=0.25)

fig.suptitle('Step 7 — Mahalanobis Distance: How "Unusual" Are Test Inputs?\n'
             '(distance from PCA-reduced training centroid — higher = more OOD)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's7_fig1_mahalanobis_profile.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s7_fig1_mahalanobis_profile.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Distance vs error for each outlier type
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 2: Distance vs error per type...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, otype, color, title in [
    (axes[0], 'Inlier', '#388E3C', 'Inlier Runs'),
    (axes[1], 'Type1',  '#D32F2F', 'Type 1 Runs (Bad XGB)'),
    (axes[2], 'Type2',  '#1976D2', 'Type 2 Runs (Divergent)')]:

    keys = [(r['experiment'], r['model']) for r in records if r['otype'] == otype]
    if not keys:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        continue
    errs = np.array([error_store[k] for k in keys if k in error_store])
    mean_err = errs.mean(axis=0)
    corr     = np.corrcoef(mah_dist, mean_err)[0, 1]

    sc = ax.scatter(mah_dist, mean_err, c=y_true, cmap='RdYlGn',
                    s=25, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='True RUL')

    # Regression line
    coef  = np.polyfit(mah_dist, mean_err, 1)
    xline = np.linspace(mah_dist.min(), mah_dist.max(), 100)
    ax.plot(xline, np.polyval(coef, xline), 'k--', linewidth=1.5,
            label=f'Trend  corr={corr:.3f}')

    ax.set_xlabel('Mahalanobis Distance', fontsize=10)
    ax.set_ylabel('Mean |Error| (cycles)', fontsize=10)
    ax.set_title(f'{title}\n(avg over {len(errs)} runs)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

fig.suptitle('Step 7 — Does Input OOD-ness Predict Prediction Error?\n'
             '(each point = one test cycle, coloured by true RUL)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's7_fig2_distance_vs_error.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s7_fig2_distance_vs_error.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Overlay: Mahalanobis distance and mean error trajectories
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 3: Distance and error overlay...")
fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()

# Mahalanobis on left axis
ax1.plot(cycles, mah_dist, color='#7B1FA2', linewidth=1.5,
         linestyle='--', label='Mahalanobis dist', alpha=0.8)
ax1.set_ylabel('Mahalanobis Distance', fontsize=10, color='#7B1FA2')
ax1.tick_params(axis='y', labelcolor='#7B1FA2')

# Errors on right axis
for otype, color, lw in [('Inlier','#388E3C',1.5),
                           ('Type1','#D32F2F',2.0),
                           ('Type2','#1976D2',2.0)]:
    keys     = [(r['experiment'], r['model']) for r in records if r['otype'] == otype]
    errs_all = np.array([error_store[k] for k in keys if k in error_store])
    if errs_all.size == 0:
        continue
    ax2.plot(cycles, errs_all.mean(axis=0), color=color, linewidth=lw,
             label=f'{otype} error', alpha=0.9)

ax2.set_ylabel('Mean |Error| (cycles)', fontsize=10)
ax1.set_xlabel('Cycle', fontsize=10)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
ax1.grid(alpha=0.2)
ax1.set_title('Step 7 — Mahalanobis Distance vs Error Trajectories\n'
              '(do peaks in distance coincide with peaks in error?)',
              fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's7_fig3_distance_error_overlay.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s7_fig3_distance_error_overlay.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Distance percentile bins: how does error vary in low/mid/high OOD?
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 4: Error by distance percentile bin...")
dist_bins  = np.percentile(mah_dist, [0, 33, 67, 100])
bin_labels = ['Low OOD\n(familiar)', 'Medium OOD', 'High OOD\n(unusual)']
bin_colors = ['#C8E6C9', '#FFF9C4', '#FFCCBC']

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, otype, title in [
    (axes[0], 'Inlier', 'Inlier Runs'),
    (axes[1], 'Type1',  'Type 1 Runs'),
    (axes[2], 'Type2',  'Type 2 Runs')]:

    keys = [(r['experiment'], r['model']) for r in records if r['otype'] == otype]
    if not keys:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        continue
    errs = np.array([error_store[k] for k in keys if k in error_store])
    if errs.size == 0:
        continue

    bin_means, bin_stds = [], []
    for b in range(3):
        lo, hi = dist_bins[b], dist_bins[b+1]
        mask   = (mah_dist >= lo) & (mah_dist <= hi)
        vals   = errs[:, mask].flatten()
        bin_means.append(vals.mean())
        bin_stds.append(vals.std())

    bars = ax.bar(range(3), bin_means, color=bin_colors,
                  edgecolor='gray', linewidth=0.8, alpha=0.9)
    ax.errorbar(range(3), bin_means, yerr=bin_stds,
                fmt='none', color='black', capsize=4, linewidth=1.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylabel('Mean |Error| (cycles)', fontsize=9)
    ax.set_title(f'{title}\n(n={len(errs)} runs)', fontsize=10)
    ax.grid(axis='y', alpha=0.25)

    for bar, mean in zip(bars, bin_means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

fig.suptitle('Step 7 — Does Higher Input OOD-ness Lead to Higher Error?\n'
             '(cycles binned by Mahalanobis distance percentile)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's7_fig4_error_by_distance_bin.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s7_fig4_error_by_distance_bin.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Per-model: correlation between Mahalanobis distance and error
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5: Correlation heatmap by model and type...")
fig, ax = plt.subplots(figsize=(10, 5))

corr_mat = np.full((len(MODEL_NAMES), 3), np.nan)
otype_list = ['Inlier', 'Type1', 'Type2']

for mi, m in enumerate(MODEL_NAMES):
    for ti, otype in enumerate(otype_list):
        keys = [(r['experiment'], m) for r in records
                if r['model'] == m and r['otype'] == otype]
        errs = [error_store[k] for k in keys if k in error_store]
        if len(errs) < 2:
            continue
        mean_err = np.array(errs).mean(axis=0)
        corr_mat[mi, ti] = np.corrcoef(mah_dist, mean_err)[0, 1]

im = ax.imshow(corr_mat.T, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson Correlation')
ax.set_xticks(range(len(MODEL_NAMES)))
ax.set_yticks(range(3))
ax.set_xticklabels(MODEL_NAMES, fontsize=10)
ax.set_yticklabels(otype_list, fontsize=10)
ax.set_title('Step 7 — Correlation: Mahalanobis Distance vs Error\n'
             '(per model per type — positive = error rises with OOD-ness)',
             fontsize=11, pad=10)

for ti in range(3):
    for mi in range(len(MODEL_NAMES)):
        v = corr_mat[mi, ti]
        if not np.isnan(v):
            ax.text(mi, ti, f'{v:.2f}', ha='center', va='center',
                    fontsize=9,
                    color='white' if abs(v) > 0.6 else 'black')
        else:
            ax.text(mi, ti, 'N/A', ha='center', va='center',
                    fontsize=8, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's7_fig5_correlation_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s7_fig5_correlation_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — ROC: can Mahalanobis distance alone predict high-error cycles?
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 6: ROC for distance as error predictor...")
ERROR_THRESHOLD = 10.0   # cycles — "high error" if |error| > 10

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, otype, title in [
    (axes[0], 'Inlier', 'Inlier'),
    (axes[1], 'Type1',  'Type 1'),
    (axes[2], 'Type2',  'Type 2')]:

    keys = [(r['experiment'], r['model']) for r in records if r['otype'] == otype]
    errs = np.array([error_store[k] for k in keys if k in error_store])
    if errs.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        continue

    # Label per cycle: 1 if mean error > threshold
    cycle_mean_err = errs.mean(axis=0)
    y_high         = (cycle_mean_err > ERROR_THRESHOLD).astype(int)

    if y_high.sum() == 0 or y_high.sum() == len(y_high):
        ax.text(0.5, 0.5, f'All cycles\nsame class\n(no ROC)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        continue

    fpr, tpr, _ = roc_curve(y_high, mah_dist)
    roc_auc     = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr, tpr, _ = roc_curve(y_high, -mah_dist)
        roc_auc     = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=COLORS[otype], linewidth=2,
            label=f'AUC = {roc_auc:.3f}')
    ax.plot([0,1],[0,1],'k--', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('FPR', fontsize=9)
    ax.set_ylabel('TPR', fontsize=9)
    ax.set_title(f'{title}\n(high error = |err| > {ERROR_THRESHOLD} cycles)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

fig.suptitle('Step 7 — Can Mahalanobis Distance Predict High-Error Cycles?\n'
             '(AUC of distance as a cycle-level unreliability detector)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's7_fig6_roc_distance_predictor.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s7_fig6_roc_distance_predictor.png")

# ── Save distance array for downstream steps ──────────────────────────────────
dist_df = pd.DataFrame({'cycle': cycles, 'true_rul': y_true,
                         'mahalanobis_dist': mah_dist})
dist_df.to_csv(os.path.join(OUT_DIR, 'mahalanobis_distances.csv'), index=False)
print(f"\n  Saved: mahalanobis_distances.csv")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  STEP 7 COMPLETE -- KEY TAKEAWAYS")
print("=" * 65)
print(f"\n  Mahalanobis distance range: [{mah_dist.min():.2f}, {mah_dist.max():.2f}]")
print(f"  Correlation with true RUL : {corr_rul:.4f}")
print()
print("  Overall correlations (Mahalanobis, mean |error|):")
for otype in ['Inlier', 'Type1', 'Type2']:
    keys     = [(r['experiment'], r['model']) for r in records if r['otype'] == otype]
    errs_all = np.array([error_store[k] for k in keys if k in error_store])
    if errs_all.size == 0:
        continue
    corr = np.corrcoef(mah_dist, errs_all.mean(axis=0))[0, 1]
    interp = ('strong' if abs(corr) > 0.6 else
              'moderate' if abs(corr) > 0.35 else 'weak')
    print(f"  {otype:<10}: corr = {corr:+.4f}  ({interp})")
print()
print("  -> Next: Step 8 -- Cycle position vs error correlation")
print(f"  -> Output files in: {OUT_DIR}/")
