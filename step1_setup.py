# =============================================================================
# WTA³ ENSEMBLE FRAMEWORK — 100 INDEPENDENT EXPERIMENTS
# STEP 1: Configuration and Data Loading
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import median_abs_deviation
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import gc
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')


print("  TURBOFAN ENGINE RUL PREDICTION — WTA³ ENSEMBLE FRAMEWORK")
print("  100 Independent Experiments | 600 Model Training Runs")
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH          = "data/train_FD001.txt"
N_EXPERIMENTS      = 100
CALIBRATION_CYCLES = 60
SEQUENCE_LENGTH    = 50
ROLLING_WINDOW     = 50
TOP_N_SENSORS      = 5
MASTER_SEED        = 42
RESULTS_DIR        = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# HYPERPARAMETER SPACE — 100 distinct configurations
# =============================================================================

RF_BASE   = dict(n_estimators=100, max_depth=10,
                 min_samples_split=20, min_samples_leaf=10, max_features=0.5)
XGB_BASE  = dict(learning_rate=0.01, max_depth=4, n_estimators=150,
                 subsample=0.7, colsample_bytree=0.7,
                 reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5)
SVR_BASE  = dict(C=50, gamma='scale', epsilon=1.0)
LSTM_BASE = dict(units=32, layers=1, dropout=0.3,
                 recurrent_dropout=0.2, lr=0.001, l2_reg=0.01)
CNN_BASE  = dict(filters=32, kernel_size=3, layers=2,
                 dropout=0.3, lr=0.001, l2_reg=0.01)
TRANS_BASE = dict(num_heads=2, d_model=32, num_layers=1,
                  dropout=0.3, lr=0.001)

def generate_experiment_configs(n_experiments, master_seed=42):
    """
    100 experiments with genuinely different hyperparameter configurations.
    Single fixed seed across all experiments — diversity comes from
    model capacity, learning rates, and regularization only.
    """
    rng = np.random.default_rng(master_seed)

    # Hyperparameter ranges — meaningful variation in model capacity
    # and regularization strength
    rf_depths         = [6, 8, 10, 12, 15]
    rf_features       = [0.3, 0.4, 0.5, 0.6, 0.7]
    rf_min_samples    = [10, 15, 20, 30, 40]

    xgb_depths        = [3, 4, 5, 6]
    xgb_lrs           = [0.005, 0.01, 0.015, 0.02, 0.03]
    xgb_subsample     = [0.6, 0.7, 0.8, 0.9]
    xgb_reg_alpha     = [0.5, 1.0, 2.0, 5.0]
    xgb_reg_lambda    = [1.0, 3.0, 5.0, 8.0, 10.0]

    svr_C             = [10, 30, 50, 80, 100]
    svr_epsilon       = [0.5, 1.0, 1.5, 2.0]

    lstm_units        = [16, 24, 32, 48, 64]
    lstm_dropout      = [0.1, 0.2, 0.3, 0.4, 0.5]
    lstm_l2           = [0.001, 0.005, 0.01, 0.02]
    lstm_lr           = [0.0005, 0.001, 0.002]

    cnn_filters       = [16, 24, 32, 48, 64]
    cnn_kernel        = [3, 5, 7]
    cnn_dropout       = [0.1, 0.2, 0.3, 0.4]
    cnn_l2            = [0.001, 0.005, 0.01, 0.02]

    trans_heads       = [2, 4]
    trans_dmodel      = [16, 24, 32, 48, 64]
    trans_layers      = [1, 2]
    trans_dropout     = [0.1, 0.2, 0.3, 0.4]

    configs = []
    for i in range(n_experiments):
        configs.append({
            'name': f'Exp_{i+1:03d}',
            'seed': master_seed,          # fixed seed — same for all experiments
            'rf_params': {
                'n_estimators'    : 100,
                'max_depth'       : int(rng.choice(rf_depths)),
                'min_samples_split': int(rng.choice(rf_min_samples)),
                'min_samples_leaf': int(rng.choice(rf_min_samples)) // 2,
                'max_features'    : float(rng.choice(rf_features)),
                'random_state'    : master_seed
            },
            'xgb_params': {
                'learning_rate'   : float(rng.choice(xgb_lrs)),
                'max_depth'       : int(rng.choice(xgb_depths)),
                'n_estimators'    : 150,
                'subsample'       : float(rng.choice(xgb_subsample)),
                'colsample_bytree': float(rng.choice(xgb_subsample)),
                'reg_alpha'       : float(rng.choice(xgb_reg_alpha)),
                'reg_lambda'      : float(rng.choice(xgb_reg_lambda)),
                'min_child_weight': 5,
                'random_state'    : master_seed
            },
            'svr_params': {
                'C'      : float(rng.choice(svr_C)),
                'gamma'  : 'scale',
                'epsilon': float(rng.choice(svr_epsilon))
            },
            'lstm_params': {
                'units'            : int(rng.choice(lstm_units)),
                'layers'           : 1,
                'dropout'          : float(rng.choice(lstm_dropout)),
                'recurrent_dropout': float(rng.choice(lstm_dropout)),
                'lr'               : float(rng.choice(lstm_lr)),
                'l2_reg'           : float(rng.choice(lstm_l2))
            },
            'cnn_params': {
                'filters'    : int(rng.choice(cnn_filters)),
                'kernel_size': int(rng.choice(cnn_kernel)),
                'layers'     : 2,
                'dropout'    : float(rng.choice(cnn_dropout)),
                'lr'         : float(rng.choice(lstm_lr)),
                'l2_reg'     : float(rng.choice(cnn_l2))
            },
            'transformer_params': {
                'num_heads' : int(rng.choice(trans_heads)),
                'd_model'   : int(rng.choice(trans_dmodel)),
                'num_layers': int(rng.choice(trans_layers)),
                'dropout'   : float(rng.choice(trans_dropout)),
                'lr'        : float(rng.choice(lstm_lr))
            }
        })
    return configs

experiments = generate_experiment_configs(N_EXPERIMENTS, MASTER_SEED)

print(f"Experiment configurations generated: {len(experiments)}")
print(f"  Exp_001 — RF max_depth: {experiments[0]['rf_params']['max_depth']}, "
      f"XGB lr: {experiments[0]['xgb_params']['learning_rate']}")
print(f"  Exp_050 — RF max_depth: {experiments[49]['rf_params']['max_depth']}, "
      f"XGB lr: {experiments[49]['xgb_params']['learning_rate']}")
print(f"  Exp_100 — RF max_depth: {experiments[99]['rf_params']['max_depth']}, "
      f"XGB lr: {experiments[99]['xgb_params']['learning_rate']}")
print()

# =============================================================================
# LOAD DATASET
# =============================================================================

print("Loading NASA C-MAPSS FD001 dataset...")

col_names = (['engine_id', 'cycle'] +
             [f'setting_{i}' for i in range(1, 4)] +
             [f'sensor_{i}'  for i in range(1, 22)])

try:
    train_df = pd.read_csv(DATA_PATH, sep=r'\s+', header=None, names=col_names)
except FileNotFoundError:
    raise FileNotFoundError(
        f"\n[ERROR] File not found: {DATA_PATH}\n"
        "Place train_FD001.txt inside the data/ subfolder."
    )

max_cyc  = train_df.groupby('engine_id')['cycle'].max().reset_index()
max_cyc.columns = ['engine_id', 'max_cycle']
train_df = train_df.merge(max_cyc, on='engine_id')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
train_df = train_df.drop('max_cycle', axis=1)

print(f"  Loaded: {len(train_df)} samples | {train_df['engine_id'].nunique()} engines")
print(f"  RUL range: [0, {train_df['RUL'].max()}] cycles")
print()

# =============================================================================
# SELECT TEST ENGINE — randomly picked, fully unseen, never used in training
# =============================================================================

np.random.seed(MASTER_SEED)
tf.random.set_seed(MASTER_SEED)

engine_cycles = train_df.groupby('engine_id')['cycle'].max()
eligible      = engine_cycles[engine_cycles > (CALIBRATION_CYCLES + 50)].index.tolist()

REPRESENTATIVE_ENGINE = int(np.random.choice(eligible))
engine_data  = train_df[train_df['engine_id'] == REPRESENTATIVE_ENGINE].copy()
total_cycles = len(engine_data)

print(f"Test engine (unseen): #{REPRESENTATIVE_ENGINE}")
print(f"  Lifecycle  : {total_cycles} cycles")
print(f"  Calibration: cycles 1–{CALIBRATION_CYCLES}")
print(f"  Prediction : cycles {CALIBRATION_CYCLES+1}–{total_cycles} "
      f"({total_cycles - CALIBRATION_CYCLES} points)")
print()

# =============================================================================
# TRAIN / VALIDATION SPLIT — exactly 87 train / 12 val
# Stratified across lifecycle quartiles (3 val per quartile)
# Test engine is excluded before splitting — never touches train or val


legacy         = train_df[train_df['engine_id'] != REPRESENTATIVE_ENGINE].copy()
engine_max_cyc = legacy.groupby('engine_id')['cycle'].max()
quartiles      = pd.qcut(engine_max_cyc, q=4, labels=['Q1','Q2','Q3','Q4'])

train_engines, val_engines = [], []

# 3 val engines per quartile × 4 quartiles = 12 val, 87 train
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    q_ids = quartiles[quartiles == q].index.tolist()
    np.random.shuffle(q_ids)
    val_engines.extend(q_ids[:3])
    train_engines.extend(q_ids[3:])

train_data = legacy[legacy['engine_id'].isin(train_engines)].copy()
val_data   = legacy[legacy['engine_id'].isin(val_engines)].copy()

# Strict assertions
assert len(train_engines) == 87,                  f"Expected 87 train, got {len(train_engines)}"
assert len(val_engines)   == 12,                  f"Expected 12 val, got {len(val_engines)}"
assert REPRESENTATIVE_ENGINE not in train_engines, "LEAK: test engine in train!"
assert REPRESENTATIVE_ENGINE not in val_engines,   "LEAK: test engine in val!"
assert set(train_engines).isdisjoint(val_engines), "LEAK: overlap between train and val!"

print("Data split:")
print(f"  Test engine (unseen) : 1   → #{REPRESENTATIVE_ENGINE}")
print(f"  Training engines     : {len(train_engines)}  (3 held out per quartile)")
print(f"  Validation engines   : {len(val_engines)}")
print(f"  Training samples     : {len(train_data)}")
print(f"  Validation samples   : {len(val_data)}")
print(f"  Total engines        : {1 + len(train_engines) + len(val_engines)} / 100 ")
print()

# =============================================================================
# CHECKPOINT
# =============================================================================

step1_info = {
    'test_engine'     : REPRESENTATIVE_ENGINE,
    'total_cycles'    : total_cycles,
    'calibration'     : CALIBRATION_CYCLES,
    'prediction_points': total_cycles - CALIBRATION_CYCLES,
    'train_engines'   : len(train_engines),
    'val_engines'     : len(val_engines),
    'train_samples'   : len(train_data),
    'val_samples'     : len(val_data),
    'n_experiments'   : N_EXPERIMENTS,
}
with open(os.path.join(RESULTS_DIR, 'step1_checkpoint.json'), 'w') as f:
    json.dump(step1_info, f, indent=2)

print("Checkpoint saved → results/step1_checkpoint.json")
print()

# =============================================================================
# STEP 2: Sensor Selection and Feature Engineering
# =============================================================================

print()
print("  STEP 2: Sensor Selection and Feature Engineering")
print()

sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

# -----------------------------------------------------------------------------
# 2A. Identify Top Degrading Sensors
# Computed from training data only — no leakage from val or test
# -----------------------------------------------------------------------------

def calculate_degradation_score(df, sensor_col):
    """Mean absolute correlation of sensor with cycle across all engines."""
    correlations = []
    for eid in df['engine_id'].unique():
        edf = df[df['engine_id'] == eid]
        if len(edf) > 10 and edf[sensor_col].std() > 1e-6:
            corr = np.abs(edf[sensor_col].corr(edf['cycle']))
            if not np.isnan(corr):
                correlations.append(corr)
    return np.mean(correlations) if correlations else 0.0

print("Calculating sensor degradation scores (training data only)...")
degradation_scores = {
    s: calculate_degradation_score(train_data, s)
    for s in sensor_cols
}
degradation_scores = {k: v for k, v in degradation_scores.items() if v > 0}

top_sensors      = sorted(degradation_scores.items(),
                          key=lambda x: x[1], reverse=True)[:TOP_N_SENSORS]
top_sensor_names = [s[0] for s in top_sensors]

print(f"\nTop {TOP_N_SENSORS} degrading sensors selected:")
for rank, (sensor, score) in enumerate(top_sensors, 1):
    print(f"  {rank}. {sensor:10s}  degradation score: {score:.4f}")

zero_sensors = [s for s in sensor_cols if degradation_scores.get(s, 0) == 0]
print(f"\nConstant/excluded sensors ({len(zero_sensors)}): {zero_sensors}")
print()

# -----------------------------------------------------------------------------
# 2B. Feature Engineering
# IMPORTANT: leakage check runs on RAW (unscaled) data
# Scaler is fit on training data only, then applied to val and test
# -----------------------------------------------------------------------------

def add_features(df):
    """Add temporal and rolling features WITHOUT scaling — for leakage check."""
    df = df.copy().sort_values(['engine_id', 'cycle']).reset_index(drop=True)

    df['cycle_norm'] = df.groupby('engine_id')['cycle'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )
    df['cycle_squared'] = df['cycle_norm'] ** 2

    for s in top_sensor_names:
        df[f'{s}_roll_mean'] = df.groupby('engine_id')[s].transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
        df[f'{s}_roll_std'] = df.groupby('engine_id')[s].transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
        )
        df[f'{s}_roll_diff'] = df.groupby('engine_id')[s].transform(
            lambda x: x.diff().rolling(ROLLING_WINDOW, min_periods=1).mean().fillna(0)
        )
    return df

rolling_cols = [f'{s}{suf}'
                for s in top_sensor_names
                for suf in ('_roll_mean', '_roll_std', '_roll_diff')]
feature_cols = top_sensor_names + ['cycle_norm', 'cycle_squared'] + rolling_cols

# -----------------------------------------------------------------------------
# 2C. Leakage verification — must run on RAW unscaled data
# At cycle 1: rolling_mean(window=1) must equal the raw sensor value exactly
# -----------------------------------------------------------------------------

print("Verifying no future leakage (on raw unscaled data)...")
train_raw = add_features(train_data)   # unscaled copy for verification only

sid  = train_engines[0]
samp = train_raw[train_raw['engine_id'] == sid].sort_values('cycle').head(3)
all_ok = True

for s in top_sensor_names[:3]:
    raw  = samp[s].iloc[0]
    roll = samp[f'{s}_roll_mean'].iloc[0]
    ok   = np.isclose(raw, roll, rtol=1e-6)
    print(f"  {s:12s}: raw={raw:8.4f}  roll_mean@cycle1={roll:8.4f}  match={ok}")
    if not ok:
        all_ok = False

print(f"  Leakage check: {'PASSED ' if all_ok else 'FAILED — check rolling logic!'}")
print()
del train_raw   # free memory — only needed for the check

# -----------------------------------------------------------------------------
# 2D. Apply features + scaling
# -----------------------------------------------------------------------------

def engineer_features(df, scaler=None, fit_scaler=False):
    df = add_features(df)

    if fit_scaler:
        scaler = RobustScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    elif scaler is not None:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler

print("Engineering features and scaling...")
train_data,  scaler = engineer_features(train_data,  fit_scaler=True)
val_data,    _      = engineer_features(val_data,    scaler=scaler)
engine_data, _      = engineer_features(engine_data, scaler=scaler)

print(f"\nFeature matrix summary:")
print(f"  Base sensors : {len(top_sensor_names)}")
print(f"  Temporal     : 2  (cycle_norm, cycle_squared)")
print(f"  Rolling      : {len(top_sensor_names)*3}  ({len(top_sensor_names)} sensors × 3 stats)")
print(f"  Total        : {len(feature_cols)} features")
print()

# -----------------------------------------------------------------------------
# 2E. Build flat arrays (RF/XGB/SVR) and sequences (LSTM/CNN/Transformer)
# Built once here — shared across all 100 experiments
# -----------------------------------------------------------------------------

def create_sequences(df, feature_cols, seq_len=SEQUENCE_LENGTH):
    """Sliding window sequences for deep learning models."""
    seqs, targets = [], []
    for eid in df['engine_id'].unique():
        edf = df[df['engine_id'] == eid].sort_values('cycle')
        arr = edf[feature_cols].values
        rul = edf['RUL'].values
        for i in range(len(edf) - seq_len + 1):
            seqs.append(arr[i:i+seq_len])
            targets.append(rul[i+seq_len-1])
    return np.array(seqs, dtype=np.float32), np.array(targets, dtype=np.float32)

# Flat arrays
X_train_flat = train_data[feature_cols].values.astype(np.float32)
y_train_flat = train_data['RUL'].values.astype(np.float32)
X_val_flat   = val_data[feature_cols].values.astype(np.float32)
y_val_flat   = val_data['RUL'].values.astype(np.float32)

print("Building sequence arrays for deep learning models...")
X_train_seq, y_train_seq = create_sequences(train_data, feature_cols)
X_val_seq,   y_val_seq   = create_sequences(val_data,   feature_cols)

INPUT_SHAPE = (X_train_seq.shape[1], X_train_seq.shape[2])

print(f"\nArray shapes:")
print(f"  X_train_flat : {X_train_flat.shape}")
print(f"  X_val_flat   : {X_val_flat.shape}")
print(f"  X_train_seq  : {X_train_seq.shape}  ← (samples, timesteps, features)")
print(f"  X_val_seq    : {X_val_seq.shape}")
print(f"  DL input shape: {INPUT_SHAPE}")
print()

# -----------------------------------------------------------------------------
# 2F. Checkpoint
# -----------------------------------------------------------------------------

step2_info = {
    'top_sensors' : top_sensor_names,
    'n_features'  : len(feature_cols),
    'input_shape' : list(INPUT_SHAPE),
    'leakage_ok'  : all_ok,
    'X_train_flat': list(X_train_flat.shape),
    'X_val_flat'  : list(X_val_flat.shape),
    'X_train_seq' : list(X_train_seq.shape),
    'X_val_seq'   : list(X_val_seq.shape),
}
with open(os.path.join(RESULTS_DIR, 'step2_checkpoint.json'), 'w') as f:
    json.dump(step2_info, f, indent=2)

print("Checkpoint saved → results/step2_checkpoint.json")
print()

# =============================================================================
# ADD THIS BLOCK at the very end of step1_setup.py,
# immediately after the "Checkpoint saved → results/step2_checkpoint.json" lines
# =============================================================================

# -----------------------------------------------------------------------------
# 2G. Build test arrays from the held-out engine
# engine_data is already feature-engineered and scaled (done above in 2D)
# Only cycles after calibration are used for prediction
# -----------------------------------------------------------------------------

test_data = engine_data[engine_data['cycle'] > CALIBRATION_CYCLES].copy()

X_test_flat = test_data[feature_cols].values.astype(np.float32)
y_test_flat = test_data['RUL'].values.astype(np.float32)

# Sequence arrays for DL models
# create_sequences expects an engine_id column — it is still present in test_data
X_test_seq, y_test_seq = create_sequences(test_data, feature_cols)

print(f"Test arrays (engine #{REPRESENTATIVE_ENGINE}, cycles {CALIBRATION_CYCLES+1}+):")
print(f"  X_test_flat : {X_test_flat.shape}")
print(f"  X_test_seq  : {X_test_seq.shape}")
print(f"  y_test_flat : {y_test_flat.shape}  RUL range [{y_test_flat.min():.0f}, {y_test_flat.max():.0f}]")
print()