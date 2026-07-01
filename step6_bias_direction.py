# =============================================================================
# step6_bias_direction.py
# Deep analysis of prediction bias direction for each outlier type.
# Key questions:
#   1. Is the bias consistent (always same direction) or random?
#   2. When does bias become detectable during the trajectory?
#   3. Does the bias flip at any degradation stage?
#   4. Can early-cycle bias serve as a warning signal?
# =============================================================================

import json
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind

plt.style.use('default')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

RESULTS_DIR      = 'results'
STEP4_DIR        = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step4')
OUT_DIR          = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step6')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES      = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
COLORS           = {'Type1': '#D32F2F', 'Type2': '#1976D2', 'Inlier': '#388E3C'}
PRED_START_CYCLE = 61
TOTAL_CYCLES     = 213
N_PRED           = TOTAL_CYCLES - PRED_START_CYCLE + 1

RUL_BINS  = [0, 20, 50, 100, 150, 999]
RUL_SHORT = ['0-20', '20-50', '50-100', '100-150', '>150']

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

y_true = np.array(predictions[0]['y_test'])
cycles = np.arange(PRED_START_CYCLE, TOTAL_CYCLES + 1)

# Build signed error arrays per (exp, model)
error_store = {}
for pred_row in predictions:
    exp = pred_row['experiment']
    for m in MODEL_NAMES:
        p = np.array(pred_row['test_predictions'][m])
        error_store[(exp, m)] = p - y_true

zone_masks = [(y_true >= lo) & (y_true < hi)
              for lo, hi in zip(RUL_BINS[:-1], RUL_BINS[1:])]

# ─── Helper: get error matrix for an otype ────────────────────────────────────
def get_error_matrix(otype, model=None):
    recs = [r for r in records if r['otype'] == otype]
    if model:
        recs = [r for r in recs if r['model'] == model]
    arrays = [error_store[(r['experiment'], r['model'])]
              for r in recs if (r['experiment'], r['model']) in error_store]
    return np.array(arrays) if arrays else np.empty((0, N_PRED))

# ═════════════════════════════════════════════════════════════════════════════
# PRINT: bias consistency
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  BIAS CONSISTENCY ANALYSIS")
print("=" * 65)

for otype in ['Inlier', 'Type1', 'Type2']:
    mat = get_error_matrix(otype)
    if mat.size == 0:
        continue
    # Per run: fraction of cycles where prediction > true (overestimate)
    over_frac  = (mat > 0).mean(axis=1)   # shape (n_runs,)
    consistent_over  = (over_frac > 0.80).sum()
    consistent_under = (over_frac < 0.20).sum()
    mixed            = len(over_frac) - consistent_over - consistent_under
    print(f"\n  {otype} ({len(mat)} runs):")
    print(f"    Consistently overestimate  (>80% cycles): "
          f"{consistent_over:3d} runs ({consistent_over/len(mat)*100:.0f}%)")
    print(f"    Consistently underestimate (<20% cycles): "
          f"{consistent_under:3d} runs ({consistent_under/len(mat)*100:.0f}%)")
    print(f"    Mixed bias                              : "
          f"{mixed:3d} runs ({mixed/len(mat)*100:.0f}%)")
    print(f"    Mean bias across all cycles            : "
          f"{mat.mean():+.2f} cycles")

# ─── When does the bias become statistically detectable? ─────────────────────
print()
print("=" * 65)
print("  EARLY WARNING: CYCLE WHERE BIAS BECOMES DETECTABLE")
print("  (first cycle where |mean_signed_error| > 10 cycles)")
print("=" * 65)
THRESHOLD = 10.0

for otype in ['Type1', 'Type2']:
    mat = get_error_matrix(otype)
    if mat.size == 0:
        continue
    mean_err = mat.mean(axis=0)
    first_c  = None
    for i, (c, e) in enumerate(zip(cycles, mean_err)):
        if abs(e) > THRESHOLD:
            first_c = c
            first_rul = y_true[i]
            break
    if first_c:
        print(f"\n  {otype}: bias first exceeds {THRESHOLD} cycles at "
              f"cycle {first_c}  (RUL ~ {first_rul:.0f} cycles remaining)")
    else:
        print(f"\n  {otype}: bias never exceeds {THRESHOLD} cycles")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Signed error distribution per type (violin)
# ═════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Signed error distribution violin...")
fig, ax = plt.subplots(figsize=(10, 5))

all_data   = []
positions  = []
vcolors    = []
tick_lbs   = []

pos = 1
for otype, color in [('Inlier','#388E3C'),('Type1','#D32F2F'),('Type2','#1976D2')]:
    mat = get_error_matrix(otype)
    if mat.size == 0:
        continue
    flat = mat.flatten()
    # clip for visibility
    flat_c = np.clip(flat, np.percentile(flat, 1), np.percentile(flat, 99))
    all_data.append(flat_c)
    positions.append(pos)
    vcolors.append(color)
    tick_lbs.append(f'{otype}\n(n={len(mat)} runs)')
    pos += 1

vp = ax.violinplot(all_data, positions=positions, showmedians=True,
                   showextrema=True, widths=0.6)
for body, color in zip(vp['bodies'], vcolors):
    body.set_facecolor(color)
    body.set_alpha(0.65)
for part in ['cmedians','cmaxes','cmins','cbars']:
    vp[part].set_color('black')
    vp[part].set_linewidth(1)

ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6,
           label='Zero bias (perfect)')
ax.set_xticks(positions)
ax.set_xticklabels(tick_lbs, fontsize=10)
ax.set_ylabel('Signed Error  (pred – true RUL)', fontsize=10)
ax.set_title('Step 6 — Signed Error Distribution by Outlier Type\n'
             '(positive = overestimate — model thinks engine has MORE life than it does)',
             fontsize=11, pad=10)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's6_fig1_signed_error_violin.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s6_fig1_signed_error_violin.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Bias trajectory: mean signed error over all 153 cycles
#            With "detection threshold" line and zone shading
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 2: Bias trajectory with detection threshold...")
zone_bg = ['#ffebee','#fff3e0','#f9fbe7','#e8f5e9','#e3f2fd']

fig, ax = plt.subplots(figsize=(13, 5))

for b, (zmask, bg) in enumerate(zip(zone_masks, zone_bg)):
    if zmask.sum() > 0:
        c_lo = cycles[zmask].min()
        c_hi = cycles[zmask].max()
        ax.axvspan(c_lo, c_hi, alpha=0.3, color=bg, zorder=0)
        mid  = (c_lo + c_hi) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 55,
                RUL_SHORT[b], ha='center', va='top', fontsize=8, color='gray')

for otype, color, lw, ls in [('Inlier','#388E3C',1.5,'-'),
                               ('Type1','#D32F2F',2.2,'-'),
                               ('Type2','#1976D2',2.0,'--')]:
    mat = get_error_matrix(otype)
    if mat.size == 0:
        continue
    mean_b = mat.mean(axis=0)
    std_b  = mat.std(axis=0)
    ax.plot(cycles, mean_b, color=color, linewidth=lw, linestyle=ls,
            label=f'{otype} (n={len(mat)})', zorder=3)
    ax.fill_between(cycles, mean_b - 0.5*std_b, mean_b + 0.5*std_b,
                    color=color, alpha=0.10, zorder=2)

ax.axhline(0,      color='black', linewidth=1.0, linestyle=':',  alpha=0.6)
ax.axhline(+THRESHOLD,  color='orange', linewidth=1.2, linestyle='--',
           alpha=0.8, label=f'+{THRESHOLD} cycle warning threshold')
ax.axhline(-THRESHOLD,  color='orange', linewidth=1.2, linestyle='--', alpha=0.8)

ax.set_xlabel('Cycle', fontsize=11)
ax.set_ylabel('Mean Signed Error  (pred – true RUL)', fontsize=10)
ax.set_title('Step 6 — Bias Trajectory Across All 153 Cycles\n'
             '(orange lines = ±10 cycle warning threshold; zone colours = RUL ranges)',
             fontsize=11, pad=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.2, zorder=1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's6_fig2_bias_trajectory.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s6_fig2_bias_trajectory.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Per-run bias consistency: histogram of "overestimate fraction"
#            (what % of cycles does each run overestimate?)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 3: Per-run overestimate fraction...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, otype, color, title in [
    (axes[0], 'Inlier', '#388E3C', 'Inlier Runs'),
    (axes[1], 'Type1',  '#D32F2F', 'Type 1 Runs (Bad)'),
    (axes[2], 'Type2',  '#1976D2', 'Type 2 Runs (Divergent)')]:

    mat = get_error_matrix(otype)
    if mat.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        continue
    over_frac = (mat > 0).mean(axis=1) * 100
    ax.hist(over_frac, bins=20, color=color, alpha=0.8, edgecolor='white')
    ax.axvline(50, color='black', linestyle='--', linewidth=1,
               label='50% (no bias)')
    ax.axvline(over_frac.mean(), color='darkred', linestyle='-', linewidth=1.5,
               label=f'Mean={over_frac.mean():.0f}%')
    ax.set_xlabel('% of cycles where prediction > true RUL', fontsize=9)
    ax.set_ylabel('Number of runs', fontsize=9)
    ax.set_title(f'{title}\n(n={len(mat)} runs)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 100)

fig.suptitle('Step 6 — Bias Consistency: What Fraction of Cycles Does Each Run Overestimate?\n'
             '(100% = always overestimates; 0% = always underestimates; 50% = unbiased)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUT_DIR, 's6_fig3_overestimate_fraction.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s6_fig3_overestimate_fraction.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Bias evolution: early (first 30 cycles) vs late (last 30 cycles)
#            Does the bias grow, shrink, or flip direction as RUL decreases?
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 4: Early vs late bias comparison...")
EARLY_N = 30   # first 30 cycles of test window (high RUL)
LATE_N  = 30   # last  30 cycles (low RUL, critical)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, otype_focus, title in [
    (axes[0], 'Type1', 'Type 1 (XGB Bad) — Bias Evolution'),
    (axes[1], 'Type2', 'Type 2 (Transformer Divergent) — Bias Evolution')]:

    for otype, color, lw in [('Inlier','#388E3C',1.5),
                               (otype_focus, COLORS[otype_focus], 2.2)]:
        mat = get_error_matrix(otype)
        if mat.size == 0:
            continue
        mean_b = mat.mean(axis=0)

        # Early window
        ax.plot(cycles[:EARLY_N], mean_b[:EARLY_N],
                color=color, linewidth=lw, label=f'{otype} early')
        # Late window
        ax.plot(cycles[-LATE_N:], mean_b[-LATE_N:],
                color=color, linewidth=lw, linestyle='--',
                label=f'{otype} late')

        # Arrow showing direction of change
        early_mean = mean_b[:EARLY_N].mean()
        late_mean  = mean_b[-LATE_N:].mean()
        print(f"  {otype:<10} early bias={early_mean:+.2f}  "
              f"late bias={late_mean:+.2f}  "
              f"change={late_mean - early_mean:+.2f}")

    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.axhline(+THRESHOLD, color='orange', linewidth=1, linestyle='--', alpha=0.7)
    ax.axhline(-THRESHOLD, color='orange', linewidth=1, linestyle='--', alpha=0.7)
    ax.set_xlabel('Cycle', fontsize=10)
    ax.set_ylabel('Mean Signed Error', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    print()

fig.suptitle('Step 6 — Early vs Late Bias: Does the Error Pattern Change Over Time?\n'
             '(solid = first 30 cycles, dashed = last 30 cycles)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's6_fig4_early_vs_late_bias.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s6_fig4_early_vs_late_bias.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Per-model bias trajectory (subplots)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5: Per-model bias trajectories...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax = axes[j]
    has_data = False

    for b, (zmask, bg) in enumerate(zip(zone_masks, zone_bg)):
        if zmask.sum() > 0:
            ax.axvspan(cycles[zmask].min(), cycles[zmask].max(),
                       alpha=0.2, color=bg, zorder=0)

    for otype, color, lw, ls in [('Inlier','#388E3C',1.5,'-'),
                                   ('Type1','#D32F2F',2.0,'-'),
                                   ('Type2','#1976D2',2.0,'--')]:
        mat = get_error_matrix(otype, model=m)
        if mat.size == 0:
            continue
        has_data = True
        mean_b = mat.mean(axis=0)
        std_b  = mat.std(axis=0)
        ax.plot(cycles, mean_b, color=color, linewidth=lw, linestyle=ls,
                label=f'{otype} (n={len(mat)})', zorder=3)
        ax.fill_between(cycles, mean_b - 0.3*std_b, mean_b + 0.3*std_b,
                        color=color, alpha=0.10)

    ax.axhline(0,      color='black',  linewidth=0.8, linestyle=':')
    ax.axhline(+10,    color='orange', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(-10,    color='orange', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_title(m, fontsize=11)
    ax.set_xlabel('Cycle', fontsize=8)
    ax.set_ylabel('Mean Signed Error', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)

fig.suptitle('Step 6 — Bias Trajectory per Model\n'
             '(orange = ±10 cycle warning; zone shading = RUL phase)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's6_fig5_bias_per_model.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s6_fig5_bias_per_model.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Cumulative bias: running sum of signed error
#            Reveals whether errors compound or cancel out over time
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 6: Cumulative bias...")
fig, ax = plt.subplots(figsize=(13, 5))

for otype, color, lw, ls in [('Inlier','#388E3C',1.5,'-'),
                               ('Type1','#D32F2F',2.2,'-'),
                               ('Type2','#1976D2',2.0,'--')]:
    mat = get_error_matrix(otype)
    if mat.size == 0:
        continue
    mean_b  = mat.mean(axis=0)
    cum_b   = np.cumsum(mean_b)
    ax.plot(cycles, cum_b, color=color, linewidth=lw, linestyle=ls,
            label=f'{otype}  final cumulative={cum_b[-1]:+.0f}', zorder=3)

ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.6)
for b, (zmask, bg) in enumerate(zip(zone_masks, zone_bg)):
    if zmask.sum() > 0:
        ax.axvspan(cycles[zmask].min(), cycles[zmask].max(),
                   alpha=0.2, color=bg, zorder=0)

ax.set_xlabel('Cycle', fontsize=11)
ax.set_ylabel('Cumulative Signed Error (sum over all cycles so far)', fontsize=10)
ax.set_title('Step 6 — Cumulative Bias: Do Errors Compound or Cancel?\n'
             '(steep slope = errors accumulating consistently in one direction)',
             fontsize=11, pad=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's6_fig6_cumulative_bias.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: s6_fig6_cumulative_bias.png")

# ── Statistical tests ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  STATISTICAL TESTS: IS THE BIAS SIGNIFICANTLY DIFFERENT?")
print("=" * 65)
for otype in ['Type1', 'Type2']:
    mat_ot = get_error_matrix(otype).flatten()
    mat_in = get_error_matrix('Inlier').flatten()
    if mat_ot.size == 0:
        continue
    # Sample for speed
    sample_size = min(5000, len(mat_ot), len(mat_in))
    rng = np.random.default_rng(42)
    s_ot = rng.choice(mat_ot, sample_size, replace=False)
    s_in = rng.choice(mat_in, sample_size, replace=False)
    t, p  = ttest_ind(s_ot, s_in)
    print(f"\n  {otype} vs Inlier:")
    print(f"    Mean bias: {otype}={mat_ot.mean():+.3f}  Inlier={mat_in.mean():+.3f}")
    print(f"    t-statistic = {t:.2f},  p-value = {p:.2e}")
    print(f"    {'** Statistically significant (p<0.001)' if p < 0.001 else 'Not significant'}")

# ── Final summary ──────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  STEP 6 COMPLETE -- KEY TAKEAWAYS")
print("=" * 65)

for otype in ['Inlier', 'Type1', 'Type2']:
    mat = get_error_matrix(otype)
    if mat.size == 0:
        continue
    over_frac   = (mat > 0).mean(axis=1) * 100
    always_over = (over_frac > 80).sum()
    mean_bias   = mat.mean()
    print(f"\n  {otype} ({len(mat)} runs):")
    print(f"    Mean bias        : {mean_bias:+.2f} cycles")
    print(f"    Consistently over: {always_over}/{len(mat)} runs "
          f"({always_over/len(mat)*100:.0f}%)")

print()
print("  -> Next: Step 7 -- Mahalanobis distance (input OOD analysis)")
print(f"  -> Output files in: {OUT_DIR}/")
