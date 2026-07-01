# =============================================================================
# step5_rul_zone_error.py
# Analyse prediction error by RUL zone (degradation phase) per outlier type.
# Key question: where in the engine's lifetime does each type fail worst,
# and is the error systematic (biased) or random (noisy)?
# =============================================================================

import json
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('default')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

RESULTS_DIR      = 'results'
STEP4_DIR        = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step4')
OUT_DIR          = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step5')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES      = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
COLORS           = {'Type1': '#D32F2F', 'Type2': '#1976D2', 'Inlier': '#388E3C'}
PRED_START_CYCLE = 61
TOTAL_CYCLES     = 213
N_PRED           = TOTAL_CYCLES - PRED_START_CYCLE + 1   # 153

RUL_BINS   = [0, 20, 50, 100, 150, 999]
RUL_LABELS = ['0–20\n(critical)', '20–50\n(late)', '50–100\n(mid)',
              '100–150\n(early)', '>150\n(very early)']
RUL_SHORT  = ['0-20', '20-50', '50-100', '100-150', '>150']

# ── Load data ─────────────────────────────────────────────────────────────────
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

rec_map = {(r['experiment'], r['model']): r for r in records}
y_true  = np.array(predictions[0]['y_test'])   # true RUL for 153 cycles
cycles  = np.arange(PRED_START_CYCLE, TOTAL_CYCLES + 1)

# Build zone mask per cycle
zone_masks = []
for lo, hi in zip(RUL_BINS[:-1], RUL_BINS[1:]):
    zone_masks.append((y_true >= lo) & (y_true < hi))

print(f"  Cycles per RUL zone: "
      f"{[m.sum() for m in zone_masks]} = {sum(m.sum() for m in zone_masks)} total\n")

# Build (exp, model) -> predictions array and signed error array
print("Building error arrays...")
pred_store = {}   # (exp, model) -> np.array shape (N_PRED,)
for pred_row in predictions:
    exp = pred_row['experiment']
    for m in MODEL_NAMES:
        pred_store[(exp, m)] = np.array(pred_row['test_predictions'][m])

# Signed error = pred - true  (positive = overestimate)
error_store = {k: v - y_true for k, v in pred_store.items()}
abs_err_store = {k: np.abs(v) for k, v in error_store.items()}

# ── Per-zone statistics ────────────────────────────────────────────────────────
print("=" * 70)
print("  MEAN ABSOLUTE ERROR BY RUL ZONE AND OUTLIER TYPE")
print("=" * 70)
print(f"\n{'RUL Zone':<12} {'Inlier MAE':>11} {'Type1 MAE':>11} "
      f"{'Type2 MAE':>11} {'T1/Inlier':>10} {'T2/Inlier':>10}")
print("-" * 60)

zone_stats = []   # for plotting
for b, (zmask, label) in enumerate(zip(zone_masks, RUL_SHORT)):
    row_s = {'zone': label, 'n_cycles': int(zmask.sum())}
    for otype in ['Inlier', 'Type1', 'Type2']:
        keys = [(r['experiment'], r['model']) for r in records if r['otype'] == otype]
        if keys:
            mae = np.mean([abs_err_store[k][zmask].mean() for k in keys
                           if k in abs_err_store])
        else:
            mae = np.nan
        row_s[f'mae_{otype}'] = mae
    zone_stats.append(row_s)
    r1 = row_s['mae_Type1'] / row_s['mae_Inlier'] if row_s['mae_Inlier'] else np.nan
    r2 = row_s['mae_Type2'] / row_s['mae_Inlier'] if row_s['mae_Inlier'] else np.nan
    print(f"  {label:<10} {row_s['mae_Inlier']:>11.2f} "
          f"{row_s['mae_Type1']:>11.2f} {row_s['mae_Type2']:>11.2f} "
          f"{r1:>10.2f}x {r2:>10.2f}x")

# Signed (bias) per zone
print(f"\n{'RUL Zone':<12} {'Inlier bias':>12} {'Type1 bias':>12} {'Type2 bias':>12}")
print("-" * 52)
for b, (zmask, label) in enumerate(zip(zone_masks, RUL_SHORT)):
    for otype in ['Inlier', 'Type1', 'Type2']:
        keys  = [(r['experiment'], r['model']) for r in records if r['otype'] == otype]
        bias  = np.mean([error_store[k][zmask].mean() for k in keys
                         if k in error_store]) if keys else np.nan
        zone_stats[b][f'bias_{otype}'] = bias
    print(f"  {label:<10} {zone_stats[b]['bias_Inlier']:>12.2f} "
          f"{zone_stats[b]['bias_Type1']:>12.2f} "
          f"{zone_stats[b]['bias_Type2']:>12.2f}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — MAE by RUL zone: grouped bar chart (Type1, Type2, Inlier)
# ═════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: MAE by RUL zone bar chart...")
fig, ax = plt.subplots(figsize=(11, 5))
x   = np.arange(len(RUL_SHORT))
w   = 0.25

for i, (otype, color) in enumerate([('Inlier','#388E3C'),
                                      ('Type1','#D32F2F'),
                                      ('Type2','#1976D2')]):
    vals = [s[f'mae_{otype}'] for s in zone_stats]
    bars = ax.bar(x + (i-1)*w, vals, w, color=color, alpha=0.85,
                  label=otype, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=7.5, color='black')

ax.set_xticks(x)
ax.set_xticklabels(RUL_LABELS, fontsize=9)
ax.set_xlabel('True RUL Zone (cycles remaining)', fontsize=11)
ax.set_ylabel('Mean Absolute Error (cycles)', fontsize=11)
ax.set_title('Step 5 — Prediction Error by Degradation Phase and Outlier Type\n'
             'How bad is each type\'s error, and at what stage?',
             fontsize=11, pad=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's5_fig1_mae_by_rul_zone.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s5_fig1_mae_by_rul_zone.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Signed bias by RUL zone (over- vs under-estimation)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 2: Signed bias by RUL zone...")
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(RUL_SHORT))
w = 0.25

for i, (otype, color) in enumerate([('Inlier','#388E3C'),
                                      ('Type1','#D32F2F'),
                                      ('Type2','#1976D2')]):
    vals = [s[f'bias_{otype}'] for s in zone_stats]
    ax.bar(x + (i-1)*w, vals, w, color=color, alpha=0.85, label=otype,
           edgecolor='white', linewidth=0.4)

ax.axhline(0, color='black', linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(RUL_LABELS, fontsize=9)
ax.set_xlabel('True RUL Zone', fontsize=11)
ax.set_ylabel('Mean Signed Error  (positive = overestimate)', fontsize=10)
ax.set_title('Step 5 — Prediction Bias Direction by Degradation Phase\n'
             'Positive = model thinks engine has MORE life than it does (dangerous)',
             fontsize=11, pad=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's5_fig2_bias_by_rul_zone.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s5_fig2_bias_by_rul_zone.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Error trajectory (cycle-by-cycle MAE) for each type
#            Full 153-cycle view with RUL zone shading
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 3: Full error trajectory with zone shading...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

zone_colors_bg = ['#ffebee','#fff3e0','#f9fbe7','#e8f5e9','#e3f2fd']

for ax, metric, title, ylabel in [
    (axes[0], 'abs', 'Mean Absolute Error Over Time',
     'Mean |Error| (cycles)'),
    (axes[1], 'signed', 'Signed Error (Bias) Over Time',
     'Mean Signed Error (cycles)')]:

    # Background zone shading
    for b, (zmask, bg) in enumerate(zip(zone_masks, zone_colors_bg)):
        if zmask.sum() > 0:
            c_lo = cycles[zmask].min()
            c_hi = cycles[zmask].max()
            ax.axvspan(c_lo, c_hi, alpha=0.35, color=bg, zorder=0)
            ax.text((c_lo+c_hi)/2, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 5,
                    RUL_SHORT[b], ha='center', va='top',
                    fontsize=7, color='gray')

    for otype, color, lw, ls in [('Inlier','#388E3C',1.5,'-'),
                                   ('Type1','#D32F2F',2.0,'-'),
                                   ('Type2','#1976D2',2.0,'--')]:
        keys = [(r['experiment'], r['model']) for r in records if r['otype'] == otype]
        if not keys:
            continue
        store = abs_err_store if metric == 'abs' else error_store
        arr   = np.array([store[k] for k in keys if k in store])
        if arr.size == 0:
            continue
        mean_c = arr.mean(axis=0)
        std_c  = arr.std(axis=0)
        ax.plot(cycles, mean_c, color=color, linewidth=lw, linestyle=ls,
                label=f'{otype} (n={len(arr)})', zorder=3)
        ax.fill_between(cycles, mean_c - 0.5*std_c, mean_c + 0.5*std_c,
                        color=color, alpha=0.10, zorder=2)

    if metric == 'signed':
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

    ax.set_xlabel('Cycle', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, zorder=1)

fig.suptitle('Step 5 — Error Trajectories Across All 153 Test Cycles\n'
             '(shaded bands = RUL zones; ribbon = ±0.5 std)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's5_fig3_error_trajectory_full.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s5_fig3_error_trajectory_full.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-model MAE by RUL zone heatmaps (Type1 and Type2 side-by-side)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 4: Per-model heatmaps by RUL zone...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, otype, title, cmap in [
    (axes[0], 'Inlier',  'Inlier',           'Greens'),
    (axes[1], 'Type1',   'Type 1 (Bad)',      'Reds'),
    (axes[2], 'Type2',   'Type 2 (Divergent)','Blues')]:

    mat = np.full((len(MODEL_NAMES), len(RUL_SHORT)), np.nan)
    for mi, m in enumerate(MODEL_NAMES):
        keys = [(r['experiment'], m) for r in records
                if r['model'] == m and r['otype'] == otype]
        if not keys:
            continue
        for b, zmask in enumerate(zone_masks):
            vals = [abs_err_store[k][zmask].mean() for k in keys
                    if k in abs_err_store and zmask.sum() > 0]
            if vals:
                mat[mi, b] = np.mean(vals)

    # Use consistent vmax across all for comparison
    vmax = np.nanmax([np.nanmax(m2) for m2 in [mat]])
    im   = ax.imshow(mat, cmap=cmap, aspect='auto',
                     vmin=0, vmax=max(vmax, 1))
    plt.colorbar(im, ax=ax, label='Mean |Error|')
    ax.set_xticks(range(len(RUL_SHORT)))
    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(RUL_SHORT, fontsize=8, rotation=15)
    ax.set_yticklabels(MODEL_NAMES, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('RUL Zone', fontsize=9)

    for mi in range(len(MODEL_NAMES)):
        for b in range(len(RUL_SHORT)):
            v = mat[mi, b]
            if not np.isnan(v):
                ax.text(b, mi, f'{v:.1f}', ha='center', va='center',
                        fontsize=8,
                        color='white' if v > vmax*0.6 else 'black')
            else:
                ax.text(b, mi, 'N/A', ha='center', va='center',
                        fontsize=7, color='gray')

fig.suptitle('Step 5 — Mean Absolute Error by Model × RUL Zone × Outlier Type\n'
             '(N/A = no runs of that type for this model)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's5_fig4_heatmap_per_model_zone.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s5_fig4_heatmap_per_model_zone.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Error ratio heatmap: Type1/Inlier and Type2/Inlier
#            "How many times worse is the outlier than a normal model?"
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5: Error ratio heatmaps...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

inlier_mat = np.full((len(MODEL_NAMES), len(RUL_SHORT)), np.nan)
for mi, m in enumerate(MODEL_NAMES):
    keys = [(r['experiment'], m) for r in records
            if r['model'] == m and r['otype'] == 'Inlier']
    for b, zmask in enumerate(zone_masks):
        vals = [abs_err_store[k][zmask].mean() for k in keys
                if k in abs_err_store and zmask.sum() > 0]
        if vals:
            inlier_mat[mi, b] = np.mean(vals)

for ax, otype, title in [
    (axes[0], 'Type1', 'Type 1 Error / Inlier Error'),
    (axes[1], 'Type2', 'Type 2 Error / Inlier Error')]:

    ratio_mat = np.full((len(MODEL_NAMES), len(RUL_SHORT)), np.nan)
    for mi, m in enumerate(MODEL_NAMES):
        keys = [(r['experiment'], m) for r in records
                if r['model'] == m and r['otype'] == otype]
        if not keys:
            continue
        for b, zmask in enumerate(zone_masks):
            vals = [abs_err_store[k][zmask].mean() for k in keys
                    if k in abs_err_store and zmask.sum() > 0]
            if vals and not np.isnan(inlier_mat[mi, b]):
                ratio_mat[mi, b] = np.mean(vals) / inlier_mat[mi, b]

    im = ax.imshow(ratio_mat, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=6)
    plt.colorbar(im, ax=ax, label='Error Ratio (outlier / inlier)')
    ax.set_xticks(range(len(RUL_SHORT)))
    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(RUL_SHORT, fontsize=8, rotation=15)
    ax.set_yticklabels(MODEL_NAMES, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('RUL Zone', fontsize=9)

    for mi in range(len(MODEL_NAMES)):
        for b in range(len(RUL_SHORT)):
            v = ratio_mat[mi, b]
            if not np.isnan(v):
                ax.text(b, mi, f'{v:.1f}x', ha='center', va='center',
                        fontsize=8,
                        color='white' if v > 4 else 'black')
            else:
                ax.text(b, mi, 'N/A', ha='center', va='center',
                        fontsize=7, color='gray')

fig.suptitle('Step 5 — How Many Times Worse Are Outliers Than Normal Models?\n'
             '(1.0x = same as inlier, >1.0x = worse, <1.0x = actually better)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's5_fig5_error_ratio_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s5_fig5_error_ratio_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Error variability: std across experiments per zone
#            Are outlier errors consistent or spread wide?
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 6: Error variability by zone...")
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(RUL_SHORT))
w = 0.25

for i, (otype, color) in enumerate([('Inlier','#388E3C'),
                                      ('Type1','#D32F2F'),
                                      ('Type2','#1976D2')]):
    means, stds = [], []
    for b, zmask in enumerate(zone_masks):
        keys  = [(r['experiment'], r['model']) for r in records if r['otype'] == otype]
        per_run = [abs_err_store[k][zmask].mean() for k in keys
                   if k in abs_err_store and zmask.sum() > 0]
        means.append(np.mean(per_run) if per_run else 0)
        stds.append(np.std(per_run)   if per_run else 0)

    bars = ax.bar(x + (i-1)*w, means, w, color=color, alpha=0.85,
                  label=otype, edgecolor='white', linewidth=0.4)
    ax.errorbar(x + (i-1)*w, means, yerr=stds, fmt='none',
                color='black', capsize=3, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(RUL_LABELS, fontsize=9)
ax.set_xlabel('True RUL Zone', fontsize=11)
ax.set_ylabel('Mean |Error| ± Std (cycles)', fontsize=10)
ax.set_title('Step 5 — Error Magnitude and Variability by Zone\n'
             '(error bars = std across experiments — wide bars = inconsistent)',
             fontsize=11, pad=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's5_fig6_error_variability.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s5_fig6_error_variability.png")

# ── Save zone stats ───────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, 'zone_error_stats.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=zone_stats[0].keys())
    writer.writeheader()
    writer.writerows(zone_stats)
print(f"\n  Saved: {csv_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  STEP 5 COMPLETE -- KEY TAKEAWAYS")
print("=" * 65)
print()
print("  Worst zone for each type (highest MAE):")
for otype in ['Type1', 'Type2', 'Inlier']:
    maes  = [s[f'mae_{otype}'] for s in zone_stats]
    worst = zone_stats[int(np.nanargmax(maes))]
    print(f"  {otype:<8}: zone {worst['zone']:>8}  MAE={worst[f'mae_{otype}']:.2f} cycles")

print()
print("  Bias direction summary:")
for otype in ['Type1', 'Type2', 'Inlier']:
    biases = [s[f'bias_{otype}'] for s in zone_stats]
    avg_b  = np.nanmean(biases)
    direct = "OVERESTIMATE (dangerous)" if avg_b > 0 else "underestimate (conservative)"
    print(f"  {otype:<8}: avg bias = {avg_b:+.2f} cycles -> {direct}")

print()
print("  -> Next: Step 6 -- Prediction bias direction (detailed)")
print(f"  -> Output files in: {OUT_DIR}/")
