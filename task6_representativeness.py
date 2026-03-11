# =============================================================================
# task6_representativeness.py
# Engine #52 Representativeness Check
#
# Question: Is Engine #52 a typical engine or an outlier?
# If it is atypically easy/hard, our results may not generalise.
#
# Checks:
#   1. Lifecycle length — where does Engine #52 sit in the fleet distribution?
#   2. Degradation rate — slope of sensor trends vs fleet
#   3. Sensor profile — mean/std of top sensors vs fleet
#   4. RUL prediction difficulty — how does a simple baseline perform on #52
#      vs all other engines? (proxy: linear regression RMSE)
#
# Output:
#   results/engine52_representativeness.json
#   results/figures/fig_engine52_representativeness.png
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/train_FD001.txt"
RESULTS_DIR  = "results"
FIGS_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "step1_checkpoint.json")) as f:
    ck1 = json.load(f)
with open(os.path.join(RESULTS_DIR, "step2_checkpoint.json")) as f:
    ck2 = json.load(f)

TEST_ENGINE      = ck1["test_engine"]          # 52
TOP_SENSORS      = ck2["top_sensors"]          # e.g. ['sensor_2', ...]
ROLLING_WINDOW   = 50

print(f"Engine representativeness check")
print(f"  Test engine  : #{TEST_ENGINE}")
print(f"  Top sensors  : {TOP_SENSORS}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
col_names = (['engine_id', 'cycle'] +
             [f'setting_{i}' for i in range(1, 4)] +
             [f'sensor_{i}'  for i in range(1, 22)])

df = pd.read_csv(DATA_PATH, sep=r'\s+', header=None, names=col_names)
max_cyc = df.groupby('engine_id')['cycle'].max().reset_index()
max_cyc.columns = ['engine_id', 'max_cycle']
df = df.merge(max_cyc, on='engine_id')
df['RUL'] = df['max_cycle'] - df['cycle']

all_engines  = df['engine_id'].unique()
n_engines    = len(all_engines)
fleet_df     = df[df['engine_id'] != TEST_ENGINE].copy()
engine52_df  = df[df['engine_id'] == TEST_ENGINE].copy()

print(f"  Total engines: {n_engines}")
print(f"  Fleet engines: {len(fleet_df['engine_id'].unique())} (excluding #52)")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Lifecycle Length
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 1: Lifecycle length...")

fleet_lengths  = fleet_df.groupby('engine_id')['cycle'].max().values
engine52_len   = engine52_df['cycle'].max()

pct_rank = float(stats.percentileofscore(fleet_lengths, engine52_len))
mean_len = float(np.mean(fleet_lengths))
std_len  = float(np.std(fleet_lengths))
z_score  = (engine52_len - mean_len) / std_len

print(f"  Engine #52 lifecycle : {engine52_len} cycles")
print(f"  Fleet mean ± std     : {mean_len:.1f} ± {std_len:.1f} cycles")
print(f"  Fleet range          : [{fleet_lengths.min()}, {fleet_lengths.max()}]")
print(f"  Percentile rank      : {pct_rank:.1f}th percentile")
print(f"  Z-score              : {z_score:.3f}")

if abs(z_score) < 0.5:
    lifecycle_verdict = "TYPICAL — within 0.5σ of fleet mean"
elif abs(z_score) < 1.0:
    lifecycle_verdict = "NEAR-TYPICAL — within 1σ of fleet mean"
elif abs(z_score) < 2.0:
    lifecycle_verdict = "MODERATE OUTLIER — 1–2σ from fleet mean"
else:
    lifecycle_verdict = "OUTLIER — >2σ from fleet mean"

print(f"  Verdict: {lifecycle_verdict}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Degradation Rate (slope of top sensors vs normalised cycle)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 2: Degradation rate (sensor slopes)...")

def get_sensor_slopes(engine_df, sensor_cols):
    """Linear slope of each sensor vs normalised cycle [0,1] per engine."""
    slopes = {}
    for eid in engine_df['engine_id'].unique():
        edf  = engine_df[engine_df['engine_id'] == eid].sort_values('cycle')
        x    = (edf['cycle'].values - edf['cycle'].min())
        xmax = x.max()
        if xmax == 0:
            continue
        x = x / xmax
        for s in sensor_cols:
            y = edf[s].values
            if np.std(y) < 1e-6:
                continue
            slope = np.polyfit(x, y, 1)[0]
            slopes.setdefault(s, []).append(slope)
    return slopes

fleet_slopes   = get_sensor_slopes(fleet_df,    TOP_SENSORS)
engine52_slopes = {}
edf52 = engine52_df.sort_values('cycle')
x52   = (edf52['cycle'].values - edf52['cycle'].min())
x52   = x52 / x52.max()
for s in TOP_SENSORS:
    y = edf52[s].values
    if np.std(y) > 1e-6:
        engine52_slopes[s] = float(np.polyfit(x52, y, 1)[0])

slope_zscores = {}
for s in TOP_SENSORS:
    if s not in fleet_slopes or s not in engine52_slopes:
        continue
    fs    = np.array(fleet_slopes[s])
    e52_s = engine52_slopes[s]
    z     = (e52_s - np.mean(fs)) / (np.std(fs) + 1e-10)
    slope_zscores[s] = float(z)
    pct   = float(stats.percentileofscore(fs, e52_s))
    print(f"  {s:12s}  fleet_mean={np.mean(fs):8.4f}  "
          f"e52={e52_s:8.4f}  z={z:6.3f}  pct={pct:.1f}th")

mean_abs_z = float(np.mean(np.abs(list(slope_zscores.values()))))
if mean_abs_z < 0.5:
    slope_verdict = "TYPICAL — degradation rate within 0.5σ across all sensors"
elif mean_abs_z < 1.0:
    slope_verdict = "NEAR-TYPICAL — degradation rate within 1σ"
else:
    slope_verdict = "ATYPICAL — degradation rate deviates >1σ"

print(f"  Mean |z| across sensors: {mean_abs_z:.3f}")
print(f"  Verdict: {slope_verdict}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Sensor Profile (mean and std of top sensors)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 3: Sensor profile (mean values)...")

sensor_profile_results = {}
for s in TOP_SENSORS:
    fleet_means = fleet_df.groupby('engine_id')[s].mean().values
    e52_mean    = float(engine52_df[s].mean())
    z           = (e52_mean - np.mean(fleet_means)) / (np.std(fleet_means) + 1e-10)
    pct         = float(stats.percentileofscore(fleet_means, e52_mean))
    sensor_profile_results[s] = {
        'fleet_mean': float(np.mean(fleet_means)),
        'fleet_std' : float(np.std(fleet_means)),
        'e52_mean'  : e52_mean,
        'z_score'   : float(z),
        'percentile': pct,
    }
    print(f"  {s:12s}  fleet={np.mean(fleet_means):8.4f}±{np.std(fleet_means):.4f}  "
          f"e52={e52_mean:8.4f}  z={z:6.3f}  pct={pct:.1f}th")

mean_profile_z = float(np.mean([abs(v['z_score'])
                                 for v in sensor_profile_results.values()]))
if mean_profile_z < 0.5:
    profile_verdict = "TYPICAL — sensor means within 0.5σ of fleet"
elif mean_profile_z < 1.0:
    profile_verdict = "NEAR-TYPICAL — sensor means within 1σ of fleet"
else:
    profile_verdict = "ATYPICAL — sensor means deviate >1σ from fleet"

print(f"  Mean |z| across sensors: {mean_profile_z:.3f}")
print(f"  Verdict: {profile_verdict}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Prediction Difficulty
# Proxy: linear regression (cycle → RUL) RMSE on each engine
# If Engine #52 is easier/harder to predict than average, results may not
# generalise. Uses only raw cycle — no feature engineering.
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 4: Prediction difficulty (linear baseline RMSE)...")

def linear_rul_rmse(engine_df, eid):
    edf = engine_df[engine_df['engine_id'] == eid].sort_values('cycle')
    x   = edf['cycle'].values.reshape(-1, 1)
    y   = edf['RUL'].values
    lr  = LinearRegression().fit(x, y)
    return float(np.sqrt(mean_squared_error(y, lr.predict(x))))

fleet_difficulty = {}
for eid in fleet_df['engine_id'].unique():
    fleet_difficulty[eid] = linear_rul_rmse(fleet_df, eid)

difficulty_vals = np.array(list(fleet_difficulty.values()))
e52_difficulty  = linear_rul_rmse(engine52_df, TEST_ENGINE)
diff_z          = (e52_difficulty - np.mean(difficulty_vals)) / np.std(difficulty_vals)
diff_pct        = float(stats.percentileofscore(difficulty_vals, e52_difficulty))

print(f"  Engine #52 linear RMSE : {e52_difficulty:.4f}")
print(f"  Fleet mean ± std       : {np.mean(difficulty_vals):.4f} ± {np.std(difficulty_vals):.4f}")
print(f"  Percentile rank        : {diff_pct:.1f}th percentile")
print(f"  Z-score                : {diff_z:.3f}")

if abs(diff_z) < 0.5:
    difficulty_verdict = "TYPICAL difficulty — linear RMSE within 0.5σ of fleet"
elif abs(diff_z) < 1.0:
    difficulty_verdict = "NEAR-TYPICAL difficulty — linear RMSE within 1σ"
else:
    difficulty_verdict = "ATYPICAL difficulty — linear RMSE >1σ from fleet mean"

print(f"  Verdict: {difficulty_verdict}")

# ─────────────────────────────────────────────────────────────────────────────
# OVERALL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
all_zscores = [abs(z_score), mean_abs_z, mean_profile_z, abs(diff_z)]
overall_z   = float(np.mean(all_zscores))

if overall_z < 0.5:
    overall_verdict = "REPRESENTATIVE — Engine #52 is typical across all checks"
elif overall_z < 1.0:
    overall_verdict = "LARGELY REPRESENTATIVE — minor deviations from fleet mean"
else:
    overall_verdict = "PARTIALLY REPRESENTATIVE — notable deviations; acknowledge in paper"

print("\n" + "="*70)
print(f"  OVERALL VERDICT: {overall_verdict}")
print(f"  Mean |z| across all checks: {overall_z:.3f}")
print("="*70)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
results = {
    'test_engine'   : TEST_ENGINE,
    'n_fleet_engines': int(len(fleet_df['engine_id'].unique())),
    'checks': {
        'lifecycle_length': {
            'engine52_cycles' : int(engine52_len),
            'fleet_mean'      : mean_len,
            'fleet_std'       : std_len,
            'fleet_min'       : int(fleet_lengths.min()),
            'fleet_max'       : int(fleet_lengths.max()),
            'percentile'      : pct_rank,
            'z_score'         : z_score,
            'verdict'         : lifecycle_verdict,
        },
        'degradation_rate': {
            'sensor_zscores'  : slope_zscores,
            'mean_abs_z'      : mean_abs_z,
            'verdict'         : slope_verdict,
        },
        'sensor_profile': {
            'per_sensor'      : sensor_profile_results,
            'mean_abs_z'      : mean_profile_z,
            'verdict'         : profile_verdict,
        },
        'prediction_difficulty': {
            'engine52_linear_rmse': e52_difficulty,
            'fleet_mean'          : float(np.mean(difficulty_vals)),
            'fleet_std'           : float(np.std(difficulty_vals)),
            'percentile'          : diff_pct,
            'z_score'             : diff_z,
            'verdict'             : difficulty_verdict,
        },
    },
    'overall': {
        'mean_abs_z_all_checks': overall_z,
        'verdict'              : overall_verdict,
    },
}

with open(os.path.join(RESULTS_DIR, 'engine52_representativeness.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: results/engine52_representativeness.json")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE — 4-panel representativeness summary
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating representativeness figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ── Panel 1: Lifecycle distribution ──────────────────────────────────────────
ax = axes[0, 0]
ax.hist(fleet_lengths, bins=20, color='#3498db', edgecolor='black',
        alpha=0.75, label='Fleet engines')
ax.axvline(engine52_len, color='#e74c3c', lw=3, ls='--',
           label=f'Engine #52 ({engine52_len} cycles)\n{pct_rank:.0f}th percentile')
ax.axvline(mean_len, color='#2c3e50', lw=1.5, ls=':',
           label=f'Fleet mean ({mean_len:.0f} cycles)')
ax.set_xlabel('Lifecycle Length (cycles)', fontsize=11, fontweight='bold')
ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('Check 1: Lifecycle Length Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel 2: Degradation rate z-scores ───────────────────────────────────────
ax = axes[0, 1]
sensors  = list(slope_zscores.keys())
zscores  = [slope_zscores[s] for s in sensors]
colors_z = ['#27ae60' if abs(z) < 1 else '#e74c3c' for z in zscores]
ax.barh(sensors, zscores, color=colors_z, edgecolor='black', alpha=0.85)
ax.axvline(0,    color='black', lw=1.5)
ax.axvline( 1.0, color='#f39c12', lw=1.5, ls='--', label='±1σ boundary')
ax.axvline(-1.0, color='#f39c12', lw=1.5, ls='--')
ax.axvline( 2.0, color='#e74c3c', lw=1,   ls=':',  label='±2σ boundary')
ax.axvline(-2.0, color='#e74c3c', lw=1,   ls=':')
ax.set_xlabel('Z-score vs Fleet', fontsize=11, fontweight='bold')
ax.set_title('Check 2: Degradation Rate Z-scores\n(|z|<1 = typical)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# ── Panel 3: Sensor profile z-scores ─────────────────────────────────────────
ax = axes[1, 0]
sensors_p  = list(sensor_profile_results.keys())
zscores_p  = [sensor_profile_results[s]['z_score'] for s in sensors_p]
colors_p   = ['#27ae60' if abs(z) < 1 else '#e74c3c' for z in zscores_p]
ax.barh(sensors_p, zscores_p, color=colors_p, edgecolor='black', alpha=0.85)
ax.axvline(0,    color='black', lw=1.5)
ax.axvline( 1.0, color='#f39c12', lw=1.5, ls='--', label='±1σ boundary')
ax.axvline(-1.0, color='#f39c12', lw=1.5, ls='--')
ax.axvline( 2.0, color='#e74c3c', lw=1,   ls=':',  label='±2σ boundary')
ax.axvline(-2.0, color='#e74c3c', lw=1,   ls=':')
ax.set_xlabel('Z-score vs Fleet', fontsize=11, fontweight='bold')
ax.set_title('Check 3: Sensor Profile Z-scores\n(|z|<1 = typical)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# ── Panel 4: Prediction difficulty distribution ───────────────────────────────
ax = axes[1, 1]
ax.hist(difficulty_vals, bins=20, color='#3498db', edgecolor='black',
        alpha=0.75, label='Fleet engines')
ax.axvline(e52_difficulty, color='#e74c3c', lw=3, ls='--',
           label=f'Engine #52 ({e52_difficulty:.2f})\n{diff_pct:.0f}th percentile')
ax.axvline(np.mean(difficulty_vals), color='#2c3e50', lw=1.5, ls=':',
           label=f'Fleet mean ({np.mean(difficulty_vals):.2f})')
ax.set_xlabel('Linear Baseline RMSE (cycles)', fontsize=11, fontweight='bold')
ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('Check 4: Prediction Difficulty\n(linear RUL baseline RMSE)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Figure: Engine #52 Fleet Representativeness\n'
             f'Overall verdict: {overall_verdict}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_engine52_representativeness.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/fig_engine52_representativeness.png")
print("\nTask 6 complete.")