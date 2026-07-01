# =============================================================================
# step1_outlier_types.py
# Separate outliers into:
#   Type 1 (Bad)       — flagged because val RMSE is too HIGH (genuinely failed)
#   Type 2 (Divergent) — flagged because val RMSE is too LOW  (different from peers)
# =============================================================================

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('default')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

RESULTS_DIR = 'results'
OUT_DIR     = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step1')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
COLORS      = {'Type1': '#D32F2F', 'Type2': '#1976D2', 'Inlier': '#388E3C'}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading training summary...")
with open(os.path.join(RESULTS_DIR, 'training_summary.json')) as f:
    training = json.load(f)
N_EXP = len(training)
print(f"  {N_EXP} experiments loaded\n")

# ── Build typed outlier table ─────────────────────────────────────────────────
# For every (experiment, model) pair, assign: Inlier / Type1 / Type2
records = []   # list of dicts — one per (exp, model)

for exp in training:
    val_rmses = np.array([exp['val_rmses'][m] for m in MODEL_NAMES])
    median_r  = np.median(val_rmses)

    for j, m in enumerate(MODEL_NAMES):
        is_outlier = exp['outliers'][m]
        vrmse      = exp['val_rmses'][m]
        trmse      = exp['train_rmses'][m]
        gap        = vrmse / trmse if trmse > 0 else 1.0

        if not is_outlier:
            otype = 'Inlier'
        elif vrmse > median_r:
            otype = 'Type1'   # bad — too high
        else:
            otype = 'Type2'   # divergent — too low

        records.append({
            'experiment': exp['experiment'],
            'model'     : m,
            'val_rmse'  : vrmse,
            'train_rmse': trmse,
            'gap_ratio' : gap,
            'median_r'  : median_r,
            'deviation' : vrmse - median_r,   # negative = below median
            'outlier'   : is_outlier,
            'otype'     : otype,
        })

# ── Summary counts ────────────────────────────────────────────────────────────
from collections import Counter

otype_counts = Counter(r['otype'] for r in records)
print("=" * 55)
print("  OVERALL OUTLIER TYPE BREAKDOWN")
print("=" * 55)
print(f"  Total (exp × model) pairs : {len(records)}")
print(f"  Inlier                    : {otype_counts['Inlier']}  "
      f"({otype_counts['Inlier']/len(records)*100:.1f}%)")
print(f"  Type 1 — Bad (high RMSE)  : {otype_counts['Type1']}  "
      f"({otype_counts['Type1']/len(records)*100:.1f}%)")
print(f"  Type 2 — Divergent (low)  : {otype_counts['Type2']}  "
      f"({otype_counts['Type2']/len(records)*100:.1f}%)")
print()

print(f"{'Model':<14} {'Inlier':>8} {'Type1':>8} {'Type2':>8}")
print("-" * 42)
for m in MODEL_NAMES:
    recs = [r for r in records if r['model'] == m]
    c    = Counter(r['otype'] for r in recs)
    print(f"  {m:<12} {c['Inlier']:>6}  {c['Type1']:>6}  {c['Type2']:>6}")
print()

# ── Per-experiment: how many of each type? ────────────────────────────────────
print("Experiments with at least one Type 1 flag:")
for exp in training:
    t1 = [m for m in MODEL_NAMES
          if exp['outliers'][m] and
          exp['val_rmses'][m] > np.median([exp['val_rmses'][mm] for mm in MODEL_NAMES])]
    if t1:
        vals = [f"{m}={exp['val_rmses'][m]:.1f}" for m in t1]
        print(f"  {exp['experiment']}  -> {', '.join(vals)}")

print()
print("Experiments with at least one Type 2 flag:")
for exp in training:
    t2 = [m for m in MODEL_NAMES
          if exp['outliers'][m] and
          exp['val_rmses'][m] < np.median([exp['val_rmses'][mm] for mm in MODEL_NAMES])]
    if t2:
        vals = [f"{m}={exp['val_rmses'][m]:.1f}" for m in t2]
        print(f"  {exp['experiment']}  -> {', '.join(vals)}")
print()

# ── Val RMSE statistics per type ──────────────────────────────────────────────
print(f"{'Category':<22} {'Mean Val RMSE':>14} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>5}")
print("-" * 70)
for otype in ['Inlier', 'Type1', 'Type2']:
    vals = [r['val_rmse'] for r in records if r['otype'] == otype]
    if vals:
        print(f"  {otype:<20} {np.mean(vals):>12.2f} {np.std(vals):>8.2f} "
              f"{np.min(vals):>8.2f} {np.max(vals):>8.2f} {len(vals):>5}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Stacked bar: type breakdown per model
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

inlier_counts = []
t1_counts     = []
t2_counts     = []

for m in MODEL_NAMES:
    recs = [r for r in records if r['model'] == m]
    c    = Counter(r['otype'] for r in recs)
    inlier_counts.append(c['Inlier'])
    t1_counts.append(c['Type1'])
    t2_counts.append(c['Type2'])

x = np.arange(len(MODEL_NAMES))
w = 0.5

b1 = ax.bar(x, inlier_counts, w, label='Inlier',            color=COLORS['Inlier'], alpha=0.85)
b2 = ax.bar(x, t1_counts,     w, label='Type 1 — Bad',      color=COLORS['Type1'],  alpha=0.85,
            bottom=inlier_counts)
b3 = ax.bar(x, t2_counts,     w, label='Type 2 — Divergent',color=COLORS['Type2'],  alpha=0.85,
            bottom=[i+t for i, t in zip(inlier_counts, t1_counts)])

# Annotate percentages on top of flagged portions
for i, (t1, t2) in enumerate(zip(t1_counts, t2_counts)):
    total_flagged = t1 + t2
    if total_flagged > 0:
        ax.text(i, 100 + 1, f'{total_flagged}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='black')

ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, fontsize=11)
ax.set_ylabel('Number of Experiments (out of 100)', fontsize=11)
ax.set_ylim(0, 115)
ax.set_title('Step 1 — Outlier Type Breakdown per Model\n'
             'Type 1 = flagged for HIGH val RMSE (bad model)  |  '
             'Type 2 = flagged for LOW val RMSE (divergent model)',
             fontsize=11, pad=12)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's1_fig1_type_breakdown_per_model.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: s1_fig1_type_breakdown_per_model.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Val RMSE distributions: Inlier vs Type1 vs Type2 (all models)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

for otype, color, zorder in [('Inlier','#388E3C',1), ('Type1','#D32F2F',3), ('Type2','#1976D2',2)]:
    vals = [r['val_rmse'] for r in records if r['otype'] == otype]
    ax.hist(vals, bins=40, alpha=0.6, color=color, label=otype,
            density=True, zorder=zorder)
    ax.axvline(np.mean(vals), color=color, linestyle='--', linewidth=1.5,
               zorder=zorder+3, label=f'{otype} mean={np.mean(vals):.1f}')

ax.set_xlabel('Validation RMSE', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Step 1 — Val RMSE Distribution by Outlier Type (all models combined)\n'
             'Type 1 clusters at HIGH RMSE, Type 2 clusters at LOW RMSE',
             fontsize=11, pad=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's1_fig2_val_rmse_by_type.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: s1_fig2_val_rmse_by_type.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Deviation from median: how far is each type from the group center?
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

for otype, color in [('Type1', '#D32F2F'), ('Type2', '#1976D2')]:
    devs = [r['deviation'] for r in records if r['otype'] == otype]
    ax.hist(devs, bins=30, alpha=0.7, color=color,
            label=f'{otype}  (mean={np.mean(devs):.1f})', density=True)

ax.axvline(0, color='black', linewidth=1.2, linestyle='-', label='Median (group center)')
ax.set_xlabel('Val RMSE − Group Median  (negative = below median, positive = above)',
              fontsize=10)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Step 1 — How Far Are Outliers From Their Group Median?\n'
             'Type 1 deviates positively (worse than peers), '
             'Type 2 deviates negatively (better than peers)',
             fontsize=11, pad=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 's1_fig3_deviation_from_median.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: s1_fig3_deviation_from_median.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-model val RMSE scatter: color by type
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for j, m in enumerate(MODEL_NAMES):
    ax   = axes[j]
    recs = [r for r in records if r['model'] == m]
    recs_sorted = sorted(recs, key=lambda r: r['val_rmse'])

    for i, r in enumerate(recs_sorted):
        color = COLORS[r['otype']]
        ax.scatter(i, r['val_rmse'], color=color, s=30, zorder=2, alpha=0.85)

    # Median line
    median_vals = np.median([r['val_rmse'] for r in recs])
    ax.axhline(median_vals, color='black', linestyle='--', linewidth=1,
               label=f'Median={median_vals:.1f}')

    ax.set_title(m, fontsize=11)
    ax.set_xlabel('Experiments (sorted by val RMSE)', fontsize=8)
    ax.set_ylabel('Val RMSE', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

# Shared legend
patches = [
    mpatches.Patch(color=COLORS['Inlier'], label='Inlier'),
    mpatches.Patch(color=COLORS['Type1'],  label='Type 1 — Bad (high RMSE)'),
    mpatches.Patch(color=COLORS['Type2'],  label='Type 2 — Divergent (low RMSE)'),
]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Step 1 — Val RMSE per Experiment, Colored by Outlier Type\n'
             '(each dot = one experiment, sorted by val RMSE)',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's1_fig4_scatter_by_type_per_model.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: s1_fig4_scatter_by_type_per_model.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Gap ratio (val/train) by outlier type
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: distribution
ax = axes[0]
for otype, color in [('Inlier','#388E3C'), ('Type1','#D32F2F'), ('Type2','#1976D2')]:
    gaps = [r['gap_ratio'] for r in records if r['otype'] == otype]
    # clip extreme gaps for visibility
    gaps_clipped = np.clip(gaps, 0, 10)
    ax.hist(gaps_clipped, bins=35, alpha=0.6, color=color, density=True,
            label=f'{otype} mean={np.mean(gaps):.2f}')
ax.axvline(1.0, color='black', linewidth=1, linestyle='--', label='Gap = 1.0 (no overfit)')
ax.set_xlabel('Gap Ratio (val RMSE / train RMSE)  [clipped at 10]', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Gap Ratio Distribution by Type', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Right: boxplot
ax = axes[1]
data_box   = []
labels_box = []
colors_box = []
for otype, color in [('Inlier','#388E3C'), ('Type1','#D32F2F'), ('Type2','#1976D2')]:
    gaps = np.clip([r['gap_ratio'] for r in records if r['otype'] == otype], 0, 10)
    data_box.append(gaps)
    labels_box.append(otype)
    colors_box.append(color)

bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, notch=False)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(1.0, color='black', linewidth=1, linestyle='--', label='Gap = 1.0')
ax.set_ylabel('Gap Ratio (clipped at 10)', fontsize=10)
ax.set_title('Gap Ratio Boxplot by Type', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Step 1 — Overfitting Gap (val/train RMSE) by Outlier Type\n'
             'Do bad models overfit? Do divergent models underfit?',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, 's1_fig5_gap_ratio_by_type.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: s1_fig5_gap_ratio_by_type.png")

# ── Save typed records for downstream steps ───────────────────────────────────
import csv

csv_path = os.path.join(OUT_DIR, 'outlier_type_labels.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
print(f"\nSaved full type-labeled table: {csv_path}")
print(f"  Columns: {list(records[0].keys())}")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  STEP 1 COMPLETE — KEY TAKEAWAYS")
print("=" * 55)
t1_total = otype_counts['Type1']
t2_total = otype_counts['Type2']
print(f"  Type 1 (Bad — high RMSE)  : {t1_total} flags across all models")
print(f"  Type 2 (Divergent — low)  : {t2_total} flags across all models")
print()

t1_models = [m for m in MODEL_NAMES
             if any(r['otype']=='Type1' for r in records if r['model']==m)]
t2_models = [m for m in MODEL_NAMES
             if any(r['otype']=='Type2' for r in records if r['model']==m)]
print(f"  Models producing Type 1 : {', '.join(t1_models) if t1_models else 'none'}")
print(f"  Models producing Type 2 : {', '.join(t2_models) if t2_models else 'none'}")
print()
print("  -> Next: Step 2 — Hyperparameter signatures")
print(f"  -> Output files in: {OUT_DIR}/")
