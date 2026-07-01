# =============================================================================
# step2_hyperparameter_signatures.py
# Find which hyperparameter values cause Type 1 (bad) and Type 2 (divergent)
# outliers. Focus: XGB (Type 1) and Transformer (Type 2).
# =============================================================================

import json
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict

plt.style.use('default')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

RESULTS_DIR  = 'results'
STEP1_DIR    = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step1')
OUT_DIR      = os.path.join(RESULTS_DIR, 'outlier_analysis', 'step2')
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {'Type1': '#D32F2F', 'Type2': '#1976D2', 'Inlier': '#388E3C'}

# ── Regenerate configs (identical logic to step1_setup.py) ───────────────────
def generate_configs(n=100, master_seed=42):
    rng = np.random.default_rng(master_seed)

    rf_depths      = [6, 8, 10, 12, 15]
    rf_features    = [0.3, 0.4, 0.5, 0.6, 0.7]
    rf_min_samples = [10, 15, 20, 30, 40]

    xgb_depths   = [3, 4, 5, 6]
    xgb_lrs      = [0.005, 0.01, 0.015, 0.02, 0.03]
    xgb_sub      = [0.6, 0.7, 0.8, 0.9]
    xgb_alpha    = [0.5, 1.0, 2.0, 5.0]
    xgb_lambda   = [1.0, 3.0, 5.0, 8.0, 10.0]

    svr_C        = [10, 30, 50, 80, 100]
    svr_eps      = [0.5, 1.0, 1.5, 2.0]

    lstm_units   = [16, 24, 32, 48, 64]
    lstm_drop    = [0.1, 0.2, 0.3, 0.4, 0.5]
    lstm_l2      = [0.001, 0.005, 0.01, 0.02]
    lstm_lr      = [0.0005, 0.001, 0.002]

    cnn_filters  = [16, 24, 32, 48, 64]
    cnn_kernel   = [3, 5, 7]
    cnn_drop     = [0.1, 0.2, 0.3, 0.4]
    cnn_l2       = [0.001, 0.005, 0.01, 0.02]

    th           = [2, 4]
    td           = [16, 24, 32, 48, 64]
    tl           = [1, 2]
    tdr          = [0.1, 0.2, 0.3, 0.4]
    tlr          = [0.0005, 0.001, 0.002]

    configs = []
    for i in range(n):
        configs.append({
            'name': f'Exp_{i+1:03d}',
            'xgb': {
                'learning_rate'   : float(rng.choice(xgb_lrs)),
                'max_depth'       : int(rng.choice(xgb_depths)),
                'subsample'       : float(rng.choice(xgb_sub)),
                'colsample_bytree': float(rng.choice(xgb_sub)),
                'reg_alpha'       : float(rng.choice(xgb_alpha)),
                'reg_lambda'      : float(rng.choice(xgb_lambda)),
            },
            'transformer': {
                'num_heads' : int(rng.choice(th)),
                'd_model'   : int(rng.choice(td)),
                'num_layers': int(rng.choice(tl)),
                'dropout'   : float(rng.choice(tdr)),
                'lr'        : float(rng.choice(tlr)),
            },
            # included for completeness but not analysed
            'rf': {
                'max_depth'        : int(rng.choice(rf_depths)),
                'min_samples_split': int(rng.choice(rf_min_samples)),
                'max_features'     : float(rng.choice(rf_features)),
            },
            'svr': {
                'C'      : float(rng.choice(svr_C)),
                'epsilon': float(rng.choice(svr_eps)),
            },
            'lstm': {
                'units'  : int(rng.choice(lstm_units)),
                'dropout': float(rng.choice(lstm_drop)),
                'l2_reg' : float(rng.choice(lstm_l2)),
                'lr'     : float(rng.choice(lstm_lr)),
            },
            'cnn': {
                'filters'    : int(rng.choice(cnn_filters)),
                'kernel_size': int(rng.choice(cnn_kernel)),
                'dropout'    : float(rng.choice(cnn_drop)),
                'l2_reg'     : float(rng.choice(cnn_l2)),
            },
        })
    return configs

print("Regenerating 100 experiment configs...")
configs = generate_configs()
config_map = {c['name']: c for c in configs}

# ── Load Step 1 type labels ───────────────────────────────────────────────────
print("Loading Step 1 outlier type labels...")
records = []
with open(os.path.join(STEP1_DIR, 'outlier_type_labels.csv')) as f:
    for row in csv.DictReader(f):
        row['val_rmse']   = float(row['val_rmse'])
        row['train_rmse'] = float(row['train_rmse'])
        row['gap_ratio']  = float(row['gap_ratio'])
        row['outlier']    = row['outlier'] == 'True'
        records.append(row)

# Index by (experiment, model) -> otype
otype_map = {(r['experiment'], r['model']): r['otype'] for r in records}
print(f"  {len(records)} records loaded\n")

# ═════════════════════════════════════════════════════════════════════════════
# XGB ANALYSIS — Type 1 (bad outliers)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("  XGB — TYPE 1 ANALYSIS")
print("=" * 55)

xgb_params   = ['learning_rate', 'max_depth', 'subsample',
                 'colsample_bytree', 'reg_alpha', 'reg_lambda']

xgb_outlier  = {}   # param -> list of values for Type1 runs
xgb_inlier   = {}   # param -> list of values for Inlier runs

for p in xgb_params:
    xgb_outlier[p] = []
    xgb_inlier[p]  = []

for exp_name, cfg in config_map.items():
    otype = otype_map.get((exp_name, 'XGB'), 'Inlier')
    for p in xgb_params:
        val = cfg['xgb'][p]
        if otype == 'Type1':
            xgb_outlier[p].append(val)
        else:
            xgb_inlier[p].append(val)

print(f"\n{'Parameter':<20} {'Outlier values':<40} {'Inlier mean':>12}")
print("-" * 75)
for p in xgb_params:
    out_vals = xgb_outlier[p]
    in_vals  = xgb_inlier[p]
    out_str  = str(sorted(set(out_vals)))
    print(f"  {p:<18} {out_str:<40} {np.mean(in_vals):>10.3f}")

# Check if any param is CONSTANT across all Type1 runs
print("\nParameters that are identical across ALL Type 1 XGB runs:")
for p in xgb_params:
    vals = xgb_outlier[p]
    if len(set(vals)) == 1:
        print(f"  ** {p} = {vals[0]}  (100% of Type 1 runs)  **")
    elif len(set(vals)) <= 2:
        c = Counter(vals)
        print(f"  {p}: {dict(c)}  ({max(c.values())/len(vals)*100:.0f}% share one value)")

# ── XGB Figure 1 — param comparison bar chart ────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, p in enumerate(xgb_params):
    ax = axes[i]

    out_vals = xgb_outlier[p]
    in_vals  = xgb_inlier[p]

    # Get all unique values for this param, sort them
    all_vals = sorted(set(out_vals + in_vals))

    out_counts = Counter(out_vals)
    in_counts  = Counter(in_vals)

    x = np.arange(len(all_vals))
    w = 0.35

    out_bars = [out_counts.get(v, 0) / len(out_vals) * 100 for v in all_vals]
    in_bars  = [in_counts.get(v, 0)  / len(in_vals)  * 100 for v in all_vals]

    ax.bar(x - w/2, out_bars, w, color=COLORS['Type1'], alpha=0.85,
           label=f'Type 1 (n={len(out_vals)})')
    ax.bar(x + w/2, in_bars,  w, color=COLORS['Inlier'], alpha=0.75,
           label=f'Inlier (n={len(in_vals)})')

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in all_vals], fontsize=8)
    ax.set_title(f'XGB: {p}', fontsize=10)
    ax.set_ylabel('% of runs', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Step 2 — XGB Hyperparameter Distribution: Type 1 vs Inlier\n'
             'Which parameter values appear exclusively in failed XGB runs?',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's2_fig1_xgb_hyperparam_type1.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: s2_fig1_xgb_hyperparam_type1.png")

# ── XGB val RMSE vs each param (scatter) ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

# Collect val_rmse per experiment for XGB
xgb_records = [r for r in records if r['model'] == 'XGB']
xgb_rmse_map = {r['experiment']: r['val_rmse'] for r in xgb_records}

for i, p in enumerate(xgb_params):
    ax = axes[i]
    xs, ys, colors_pts = [], [], []
    for exp_name, cfg in config_map.items():
        v    = cfg['xgb'][p]
        rmse = xgb_rmse_map.get(exp_name, np.nan)
        ot   = otype_map.get((exp_name, 'XGB'), 'Inlier')
        xs.append(v)
        ys.append(rmse)
        colors_pts.append(COLORS[ot])

    ax.scatter(xs, ys, c=colors_pts, s=35, alpha=0.8, zorder=2)
    ax.set_xlabel(p, fontsize=9)
    ax.set_ylabel('Val RMSE', fontsize=9)
    ax.set_title(f'XGB: val RMSE vs {p}', fontsize=10)
    ax.grid(alpha=0.2)

patches = [mpatches.Patch(color=COLORS['Type1'], label='Type 1 — Bad'),
           mpatches.Patch(color=COLORS['Inlier'], label='Inlier')]
fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Step 2 — XGB Val RMSE vs Each Hyperparameter\n'
             'Red dots (Type 1 failures) — are they concentrated at specific values?',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's2_fig2_xgb_rmse_vs_params.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: s2_fig2_xgb_rmse_vs_params.png")

# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORMER ANALYSIS — Type 2 (divergent outliers)
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 55)
print("  TRANSFORMER — TYPE 2 ANALYSIS")
print("=" * 55)

trans_params = ['num_heads', 'd_model', 'num_layers', 'dropout', 'lr']

trans_outlier = {p: [] for p in trans_params}
trans_inlier  = {p: [] for p in trans_params}

for exp_name, cfg in config_map.items():
    otype = otype_map.get((exp_name, 'Transformer'), 'Inlier')
    for p in trans_params:
        val = cfg['transformer'][p]
        if otype == 'Type2':
            trans_outlier[p].append(val)
        else:
            trans_inlier[p].append(val)

print(f"\n{'Parameter':<14} {'Type2 distribution':<45} {'Inlier distribution'}")
print("-" * 85)
for p in trans_params:
    out_c = Counter(trans_outlier[p])
    in_c  = Counter(trans_inlier[p])
    print(f"  {p:<12} {str(dict(sorted(out_c.items()))):<45} {dict(sorted(in_c.items()))}")

print("\nParameters with strong imbalance (Type2 vs Inlier %):")
for p in trans_params:
    out_vals = trans_outlier[p]
    in_vals  = trans_inlier[p]
    all_vals = sorted(set(out_vals + in_vals))
    for v in all_vals:
        out_pct = out_vals.count(v) / len(out_vals) * 100 if out_vals else 0
        in_pct  = in_vals.count(v)  / len(in_vals)  * 100 if in_vals  else 0
        diff    = out_pct - in_pct
        if abs(diff) > 10:
            direction = "OVER-represented in Type2" if diff > 0 else "UNDER-represented in Type2"
            print(f"  {p}={v}: Type2={out_pct:.0f}%  Inlier={in_pct:.0f}%  diff={diff:+.0f}%  -> {direction}")

# ── Transformer Figure 1 — param comparison bar chart ────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for i, p in enumerate(trans_params):
    ax = axes[i]
    all_vals  = sorted(set(trans_outlier[p] + trans_inlier[p]))
    out_c     = Counter(trans_outlier[p])
    in_c      = Counter(trans_inlier[p])
    x         = np.arange(len(all_vals))
    w         = 0.35
    out_bars  = [out_c.get(v, 0) / len(trans_outlier[p]) * 100 for v in all_vals]
    in_bars   = [in_c.get(v, 0)  / len(trans_inlier[p])  * 100 for v in all_vals]

    ax.bar(x - w/2, out_bars, w, color=COLORS['Type2'],  alpha=0.85,
           label=f'Type 2 (n={len(trans_outlier[p])})')
    ax.bar(x + w/2, in_bars,  w, color=COLORS['Inlier'], alpha=0.75,
           label=f'Inlier (n={len(trans_inlier[p])})')
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in all_vals], fontsize=9)
    ax.set_title(f'Transformer: {p}', fontsize=10)
    ax.set_ylabel('% of runs', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

# Hide unused 6th panel
axes[5].set_visible(False)

fig.suptitle('Step 2 — Transformer Hyperparameter Distribution: Type 2 vs Inlier\n'
             'Which configs make Transformer "too good" and get it silenced by MAD?',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's2_fig3_transformer_hyperparam_type2.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: s2_fig3_transformer_hyperparam_type2.png")

# ── Transformer val RMSE vs each param ───────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

trans_records = [r for r in records if r['model'] == 'Transformer']
trans_rmse_map = {r['experiment']: r['val_rmse'] for r in trans_records}

for i, p in enumerate(trans_params):
    ax = axes[i]
    xs, ys, colors_pts = [], [], []
    for exp_name, cfg in config_map.items():
        v    = cfg['transformer'][p]
        rmse = trans_rmse_map.get(exp_name, np.nan)
        ot   = otype_map.get((exp_name, 'Transformer'), 'Inlier')
        xs.append(v)
        ys.append(rmse)
        colors_pts.append(COLORS[ot])
    ax.scatter(xs, ys, c=colors_pts, s=35, alpha=0.8, zorder=2)
    ax.set_xlabel(p, fontsize=9)
    ax.set_ylabel('Val RMSE', fontsize=9)
    ax.set_title(f'Transformer: val RMSE vs {p}', fontsize=10)
    ax.grid(alpha=0.2)

axes[5].set_visible(False)
patches = [mpatches.Patch(color=COLORS['Type2'],  label='Type 2 — Divergent'),
           mpatches.Patch(color=COLORS['Inlier'],  label='Inlier')]
fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Step 2 — Transformer Val RMSE vs Each Hyperparameter\n'
             'Blue = Type 2 (flagged for low RMSE) — where do they cluster?',
             fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's2_fig4_transformer_rmse_vs_params.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: s2_fig4_transformer_rmse_vs_params.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Co-occurrence heatmap: pairs of Transformer params in Type2 runs
# Does a specific COMBINATION trigger Type 2?
# ═════════════════════════════════════════════════════════════════════════════
print("\nBuilding Transformer parameter co-occurrence table...")

# Focus on num_heads × d_model — most likely interaction
focus_pairs = [('num_heads', 'd_model'), ('num_heads', 'lr'),
               ('d_model', 'lr'), ('num_layers', 'd_model')]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for idx, (p1, p2) in enumerate(focus_pairs):
    ax = axes[idx]

    all_p1 = sorted(set(cfg['transformer'][p1] for cfg in config_map.values()))
    all_p2 = sorted(set(cfg['transformer'][p2] for cfg in config_map.values()))

    # Count Type2 and total per (p1, p2) cell
    count_type2 = defaultdict(int)
    count_total = defaultdict(int)

    for exp_name, cfg in config_map.items():
        v1   = cfg['transformer'][p1]
        v2   = cfg['transformer'][p2]
        ot   = otype_map.get((exp_name, 'Transformer'), 'Inlier')
        count_total[(v1, v2)] += 1
        if ot == 'Type2':
            count_type2[(v1, v2)] += 1

    # Build rate matrix
    rate_matrix = np.zeros((len(all_p1), len(all_p2)))
    for i, v1 in enumerate(all_p1):
        for j, v2 in enumerate(all_p2):
            total = count_total[(v1, v2)]
            rate_matrix[i, j] = count_type2[(v1, v2)] / total * 100 if total > 0 else 0

    im = ax.imshow(rate_matrix, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=100)
    ax.set_xticks(range(len(all_p2)))
    ax.set_yticks(range(len(all_p1)))
    ax.set_xticklabels([str(v) for v in all_p2], fontsize=9)
    ax.set_yticklabels([str(v) for v in all_p1], fontsize=9)
    ax.set_xlabel(p2, fontsize=10)
    ax.set_ylabel(p1, fontsize=10)
    ax.set_title(f'{p1} x {p2}', fontsize=10)
    plt.colorbar(im, ax=ax, label='% Type 2 runs')

    for i in range(len(all_p1)):
        for j in range(len(all_p2)):
            total = count_total[(all_p1[i], all_p2[j])]
            t2    = count_type2[(all_p1[i], all_p2[j])]
            if total > 0:
                ax.text(j, i, f'{t2}/{total}', ha='center', va='center',
                        fontsize=8,
                        color='white' if rate_matrix[i,j] > 50 else 'black')

fig.suptitle('Step 2 — Transformer: Type 2 Rate by Hyperparameter Combinations\n'
             'Cell value = (Type2 count) / (total runs with that combo) — '
             'dark = high failure rate',
             fontsize=11, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 's2_fig5_transformer_param_cooccurrence.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: s2_fig5_transformer_param_cooccurrence.png")

# ── Save hyperparameter table for downstream steps ────────────────────────────
rows = []
for exp_name, cfg in config_map.items():
    for model in ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']:
        ot = otype_map.get((exp_name, model), 'Inlier')
        model_key = model.lower().replace('transformer', 'transformer')
        if model == 'XGB':       params = cfg['xgb']
        elif model == 'Transformer': params = cfg['transformer']
        elif model == 'RF':      params = cfg['rf']
        elif model == 'SVR':     params = cfg['svr']
        elif model == 'LSTM':    params = cfg['lstm']
        else:                    params = cfg['cnn']
        row = {'experiment': exp_name, 'model': model, 'otype': ot}
        row.update({f'param_{k}': v for k, v in params.items()})
        rows.append(row)

csv_path = os.path.join(OUT_DIR, 'hyperparam_with_types.csv')
all_keys = []
for row in rows:
    for k in row.keys():
        if k not in all_keys:
            all_keys.append(k)
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore',
                            restval='')
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved: {csv_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  STEP 2 COMPLETE — KEY TAKEAWAYS")
print("=" * 55)
print()
print("XGB Type 1:")
for p in xgb_params:
    vals = xgb_outlier[p]
    if vals:
        c = Counter(vals)
        dominant = c.most_common(1)[0]
        print(f"  {p}: most common in failures = {dominant[0]} "
              f"({dominant[1]/len(vals)*100:.0f}% of Type1 runs)")

print()
print("Transformer Type 2:")
for p in trans_params:
    vals = trans_outlier[p]
    if vals:
        c = Counter(vals)
        dominant = c.most_common(1)[0]
        in_c = Counter(trans_inlier[p])
        in_pct = in_c.get(dominant[0], 0) / len(trans_inlier[p]) * 100
        out_pct = dominant[1] / len(vals) * 100
        print(f"  {p}: dominant in Type2 = {dominant[0]} "
              f"({out_pct:.0f}% of Type2  vs  {in_pct:.0f}% of Inlier)")

print()
print("-> Next: Step 3 -- Training dynamics (overfitting gap analysis)")
print(f"-> Output files in: {OUT_DIR}/")
