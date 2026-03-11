# =============================================================================
# step5_evaluation_v3.py
# KEY FIX: CI calibration now uses VALIDATION residuals only.
#          Validation residuals and per-cycle std saved to
#          results/val_calibration_data.json for step6 to consume.
#
# Changes from v2:
#   1. Collect val residuals and within-config model std per cycle
#   2. Save val_calibration_data.json
#   3. Everything else identical to v2
# =============================================================================
import time
import numpy as np
import pandas as pd
import json
import os
import gc
import warnings
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

from step3_models import build_model

# ── Constants — must match step1_setup.py exactly ────────────────────────────
DATA_PATH          = "data/train_FD001.txt"
RESULTS_DIR        = "results"
N_EXPERIMENTS      = 100
MASTER_SEED        = 42
CALIBRATION_CYCLES = 60
SEQUENCE_LENGTH    = 50
ROLLING_WINDOW     = 50
MODEL_NAMES        = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']
PRED_START_CYCLE   = CALIBRATION_CYCLES + 1   # 61
TOTAL_CYCLES       = 213
N_PRED             = TOTAL_CYCLES - PRED_START_CYCLE + 1  # 153

# ── Reconstruct all data arrays from raw file + checkpoints ──────────────────
def _load_arrays():
    with open(os.path.join(RESULTS_DIR, 'step1_checkpoint.json')) as f:
        ck1 = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'step2_checkpoint.json')) as f:
        ck2 = json.load(f)

    REPRESENTATIVE_ENGINE = ck1['test_engine']
    top_sensor_names      = ck2['top_sensors']

    col_names = (['engine_id', 'cycle'] +
                 [f'setting_{i}' for i in range(1, 4)] +
                 [f'sensor_{i}'  for i in range(1, 22)])
    df = pd.read_csv(DATA_PATH, sep=r'\s+', header=None, names=col_names)
    max_cyc = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cyc.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cyc, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop('max_cycle', axis=1)

    # Identical split
    np.random.seed(MASTER_SEED)
    legacy         = df[df['engine_id'] != REPRESENTATIVE_ENGINE].copy()
    engine_max_cyc = legacy.groupby('engine_id')['cycle'].max()
    quartiles      = pd.qcut(engine_max_cyc, q=4, labels=['Q1','Q2','Q3','Q4'])
    train_engines, val_engines = [], []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_ids = quartiles[quartiles == q].index.tolist()
        np.random.shuffle(q_ids)
        val_engines.extend(q_ids[:3])
        train_engines.extend(q_ids[3:])

    assert len(train_engines) == 87
    assert len(val_engines)   == 12
    assert REPRESENTATIVE_ENGINE not in train_engines
    assert REPRESENTATIVE_ENGINE not in val_engines

    train_data  = legacy[legacy['engine_id'].isin(train_engines)].copy()
    val_data    = legacy[legacy['engine_id'].isin(val_engines)].copy()
    engine_data = df[df['engine_id'] == REPRESENTATIVE_ENGINE].copy()

    rolling_cols = [f'{s}{suf}'
                    for s in top_sensor_names
                    for suf in ('_roll_mean', '_roll_std', '_roll_diff')]
    feature_cols = top_sensor_names + ['cycle_norm', 'cycle_squared'] + rolling_cols

    def add_features(d):
        d = d.copy().sort_values(['engine_id', 'cycle']).reset_index(drop=True)
        d['cycle_norm'] = d.groupby('engine_id')['cycle'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
        d['cycle_squared'] = d['cycle_norm'] ** 2
        for s in top_sensor_names:
            d[f'{s}_roll_mean'] = d.groupby('engine_id')[s].transform(
                lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
            d[f'{s}_roll_std']  = d.groupby('engine_id')[s].transform(
                lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))
            d[f'{s}_roll_diff'] = d.groupby('engine_id')[s].transform(
                lambda x: x.diff().rolling(ROLLING_WINDOW, min_periods=1).mean().fillna(0))
        return d

    train_data  = add_features(train_data)
    val_data    = add_features(val_data)
    engine_data = add_features(engine_data)

    # Fit scaler on training data only
    scaler = RobustScaler()
    train_data[feature_cols]  = scaler.fit_transform(train_data[feature_cols])
    val_data[feature_cols]    = scaler.transform(val_data[feature_cols])
    engine_data[feature_cols] = scaler.transform(engine_data[feature_cols])

    def make_seqs(d):
        seqs, tgts = [], []
        for eid in d['engine_id'].unique():
            edf = d[d['engine_id'] == eid].sort_values('cycle')
            arr = edf[feature_cols].values
            rul = edf['RUL'].values
            for i in range(len(edf) - SEQUENCE_LENGTH + 1):
                seqs.append(arr[i:i+SEQUENCE_LENGTH])
                tgts.append(rul[i+SEQUENCE_LENGTH-1])
        return np.array(seqs, dtype=np.float32), np.array(tgts, dtype=np.float32)

    # ── Val sequences with engine_id tracking for per-engine residuals ────────
    # We need val predictions aligned to y_val_flat for residual collection.
    # DL models produce seq outputs — we tail-align to flat as in v2.
    X_train_flat = train_data[feature_cols].values.astype(np.float32)
    y_train_flat = train_data['RUL'].values.astype(np.float32)
    X_val_flat   = val_data[feature_cols].values.astype(np.float32)
    y_val_flat   = val_data['RUL'].values.astype(np.float32)
    X_train_seq, y_train_seq = make_seqs(train_data)
    X_val_seq,   y_val_seq   = make_seqs(val_data)

    # ── Test arrays ───────────────────────────────────────────────────────────
    test_data   = engine_data[engine_data['cycle'] >= PRED_START_CYCLE].copy()
    X_test_flat = test_data[feature_cols].values.astype(np.float32)
    y_test_flat = test_data['RUL'].values.astype(np.float32)

    engine_sorted = engine_data.sort_values('cycle').reset_index(drop=True)
    feat_array    = engine_sorted[feature_cols].values.astype(np.float32)
    rul_array     = engine_sorted['RUL'].values.astype(np.float32)

    X_test_seq_list, y_test_seq_list = [], []
    for c in range(PRED_START_CYCLE, TOTAL_CYCLES + 1):
        idx     = c - 1
        history = feat_array[:idx+1]
        if len(history) >= SEQUENCE_LENGTH:
            seq = history[-SEQUENCE_LENGTH:]
        else:
            n_pad = SEQUENCE_LENGTH - len(history)
            pad   = np.repeat(history[0:1], n_pad, axis=0)
            seq   = np.vstack([pad, history])
        assert seq.shape == (SEQUENCE_LENGTH, len(feature_cols))
        X_test_seq_list.append(seq)
        y_test_seq_list.append(rul_array[idx])

    X_test_seq = np.array(X_test_seq_list, dtype=np.float32)
    y_test_seq = np.array(y_test_seq_list, dtype=np.float32)

    INPUT_SHAPE = (X_train_seq.shape[1], X_train_seq.shape[2])

    print(f"Data reconstructed  (engine #{REPRESENTATIVE_ENGINE})")
    print(f"  train flat {X_train_flat.shape}  val flat {X_val_flat.shape}")
    print(f"  test flat  {X_test_flat.shape}   cycles {PRED_START_CYCLE}–{TOTAL_CYCLES}")
    print(f"  test seq   {X_test_seq.shape}    padded for 61–109")
    print(f"  y_test RUL range: [{y_test_flat.min():.0f}, {y_test_flat.max():.0f}]")
    print()

    return (feature_cols, INPUT_SHAPE,
            X_train_flat, y_train_flat,
            X_val_flat,   y_val_flat,
            X_train_seq,  y_train_seq,
            X_val_seq,    y_val_seq,
            X_test_flat,  y_test_flat,
            X_test_seq,   y_test_seq)


def nasa_score(y_true, y_pred):
    e = np.array(y_pred) - np.array(y_true)
    s = np.where(e < 0, np.exp(-e/13.0)-1, np.exp(e/10.0)-1)
    return float(np.sum(s))


# ── Single experiment evaluation ──────────────────────────────────────────────
def evaluate_experiment(config, exp_idx, total, arrays, INPUT_SHAPE):
    (feature_cols,  _,
     X_train_flat, y_train_flat,
     X_val_flat,   y_val_flat,
     X_train_seq,  y_train_seq,
     X_val_seq,    y_val_seq,
     X_test_flat,  y_test_flat,
     X_test_seq,   y_test_seq) = arrays

    name = config['name']
    seed = config['seed']
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print(f"  Evaluating {exp_idx+1:03d}/{total}  [{name}]")

    with open(os.path.join(RESULTS_DIR, 'training_summary.json')) as f:
        summary = json.load(f)
    weights = summary[exp_idx]['weights']

    model_params = {
        'RF'         : config['rf_params'],
        'XGB'        : config['xgb_params'],
        'SVR'        : config['svr_params'],
        'LSTM'       : config['lstm_params'],
        'CNN'        : config['cnn_params'],
        'Transformer': config['transformer_params'],
    }

    test_preds   = {}
    val_preds    = {}
    train_times  = {}   # seconds per model
    infer_times  = {}   # seconds per prediction cycle (test inference / N_PRED)

    for mname in MODEL_NAMES:
        params = model_params[mname]
        model  = build_model(mname, params, INPUT_SHAPE)

        if mname in ('RF', 'XGB', 'SVR'):
            # ── Training time ────────────────────────────────────────────────
            t_train_start = time.perf_counter()
            if mname == 'XGB':
                model.fit(X_train_flat, y_train_flat,
                          eval_set=[(X_val_flat, y_val_flat)],
                          verbose=False)
            else:
                model.fit(X_train_flat, y_train_flat)
            train_times[mname] = time.perf_counter() - t_train_start

            # ── Inference time ───────────────────────────────────────────────
            t_infer_start = time.perf_counter()
            test_preds[mname] = model.predict(X_test_flat).tolist()
            infer_times[mname] = (time.perf_counter() - t_infer_start) / N_PRED

            val_preds[mname]  = model.predict(X_val_flat).tolist()

        else:
            cb = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, min_delta=0.1,
                restore_best_weights=True, verbose=0)

            # ── Training time ────────────────────────────────────────────────
            t_train_start = time.perf_counter()
            model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50, batch_size=64,
                callbacks=[cb], verbose=0
            )
            train_times[mname] = time.perf_counter() - t_train_start

            # ── Inference time ───────────────────────────────────────────────
            t_infer_start = time.perf_counter()
            test_preds[mname] = model.predict(X_test_seq, verbose=0).flatten().tolist()
            infer_times[mname] = (time.perf_counter() - t_infer_start) / N_PRED

            val_preds[mname]  = model.predict(X_val_seq, verbose=0).flatten().tolist()

        tf.keras.backend.clear_session()


    tp = {m: np.array(test_preds[m]) for m in MODEL_NAMES}
    vp = {m: np.array(val_preds[m])  for m in MODEL_NAMES}

    for m in MODEL_NAMES:
        assert len(tp[m]) == N_PRED, f"{m} test pred {len(tp[m])} != {N_PRED}"

    # Tail-align val predictions (DL produces fewer than flat)
    min_val   = min(len(vp[m]) for m in MODEL_NAMES)
    y_val_arr = y_val_flat[-min_val:]
    for m in MODEL_NAMES:
        vp[m] = vp[m][-min_val:]

    # ── WTA3 ensemble predictions ─────────────────────────────────────────────
    ens_test = sum(weights[m] * tp[m] for m in MODEL_NAMES)
    ens_val  = sum(weights[m] * vp[m] for m in MODEL_NAMES)

    # ── FIXED: collect validation residuals and within-config model std ───────
    # Residual per val cycle: |y_val - ens_val|
    val_residuals = np.abs(y_val_arr - ens_val).tolist()   # (min_val,)

    # Within-config model spread at each val cycle: std across inlier models
    # Use only models with non-zero weight (inliers)
    inlier_val_preds = np.array([
        vp[m] for m in MODEL_NAMES if weights[m] > 0
    ])  # (n_inliers, min_val)

    if inlier_val_preds.shape[0] > 1:
        val_within_std = np.std(inlier_val_preds, axis=0).tolist()   # (min_val,)
    else:
        val_within_std = np.zeros(min_val).tolist()

    # ── Metrics ───────────────────────────────────────────────────────────────
    def compute_metrics(y_true, y_pred):
        rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae   = float(mean_absolute_error(y_true, y_pred))
        score = nasa_score(y_true, y_pred)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2    = float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0
        return {'rmse': rmse, 'mae': mae, 'nasa_score': score, 'r2': r2}

    ens_test_metrics = compute_metrics(y_test_flat, ens_test)
    ens_val_metrics  = compute_metrics(y_val_arr,   ens_val)
    per_model_test   = {m: compute_metrics(y_test_flat, tp[m]) for m in MODEL_NAMES}
    per_model_val    = {m: compute_metrics(y_val_arr,   vp[m]) for m in MODEL_NAMES}

    print(f"    Ensemble  RMSE={ens_test_metrics['rmse']:.2f}  "
          f"MAE={ens_test_metrics['mae']:.2f}  "
          f"R²={ens_test_metrics['r2']:.4f}  "
          f"val_res_mean={np.mean(val_residuals):.2f}")

    return {
        'experiment'        : name,
        'seed'              : seed,
        'weights'           : weights,
        'ensemble_test'     : ens_test_metrics,
        'ensemble_val'      : ens_val_metrics,
        'per_model_test'    : per_model_test,
        'per_model_val'     : per_model_val,
        'test_predictions'  : {m: tp[m].tolist() for m in MODEL_NAMES},
        'ensemble_test_pred': ens_test.tolist(),
        'y_test'            : y_test_flat.tolist(),
        # ── NEW: validation calibration data ──────────────────────────────────
        'val_residuals'     : val_residuals,      # |y_val - ens_val| per cycle
        'val_within_std'    : val_within_std,     # std across inlier models per val cycle
        'n_val_cycles'      : min_val,
        'train_times'  : train_times,    # {model: seconds}
        'infer_times'  : infer_times,    # {model: seconds per cycle}
    }


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_all_evaluations():
    print("=" * 70)
    print("  STEP 5 v3: Evaluating 100 Experiments")
    print(f"  N_PRED={N_PRED} points  cycles {PRED_START_CYCLE}–{TOTAL_CYCLES}")
    print("  KEY FIX: CI calibration uses validation residuals only")
    print("=" * 70)
    print()

    arrays = _load_arrays()
    feature_cols, INPUT_SHAPE = arrays[0], arrays[1]

    # Re-generate identical configs
    def _gen_configs(n, seed=42):
        rng = np.random.default_rng(seed)
        rf_d=[6,8,10,12,15]; rf_f=[0.3,0.4,0.5,0.6,0.7]; rf_m=[10,15,20,30,40]
        xd=[3,4,5,6]; xl=[0.005,0.01,0.015,0.02,0.03]; xs=[0.6,0.7,0.8,0.9]
        xa=[0.5,1.0,2.0,5.0]; xr=[1.0,3.0,5.0,8.0,10.0]
        sc=[10,30,50,80,100]; se=[0.5,1.0,1.5,2.0]
        lu=[16,24,32,48,64]; ld=[0.1,0.2,0.3,0.4,0.5]
        ll=[0.001,0.005,0.01,0.02]; lr=[0.0005,0.001,0.002]
        cf=[16,24,32,48,64]; ck=[3,5,7]; cd=[0.1,0.2,0.3,0.4]
        th=[2,4]; td=[16,24,32,48,64]; tl=[1,2]; tdr=[0.1,0.2,0.3,0.4]
        out=[]
        for i in range(n):
            out.append({'name':f'Exp_{i+1:03d}','seed':seed,
                'rf_params':{'n_estimators':100,
                    'max_depth':int(rng.choice(rf_d)),
                    'min_samples_split':int(rng.choice(rf_m)),
                    'min_samples_leaf':int(rng.choice(rf_m))//2,
                    'max_features':float(rng.choice(rf_f)),
                    'random_state':seed},
                'xgb_params':{'learning_rate':float(rng.choice(xl)),
                    'max_depth':int(rng.choice(xd)),'n_estimators':150,
                    'subsample':float(rng.choice(xs)),
                    'colsample_bytree':float(rng.choice(xs)),
                    'reg_alpha':float(rng.choice(xa)),
                    'reg_lambda':float(rng.choice(xr)),
                    'min_child_weight':5,'random_state':seed},
                'svr_params':{'C':float(rng.choice(sc)),
                    'gamma':'scale','epsilon':float(rng.choice(se))},
                'lstm_params':{'units':int(rng.choice(lu)),'layers':1,
                    'dropout':float(rng.choice(ld)),
                    'recurrent_dropout':float(rng.choice(ld)),
                    'lr':float(rng.choice(lr)),'l2_reg':float(rng.choice(ll))},
                'cnn_params':{'filters':int(rng.choice(cf)),
                    'kernel_size':int(rng.choice(ck)),'layers':2,
                    'dropout':float(rng.choice(cd)),
                    'lr':float(rng.choice(lr)),'l2_reg':float(rng.choice(ll))},
                'transformer_params':{'num_heads':int(rng.choice(th)),
                    'd_model':int(rng.choice(td)),
                    'num_layers':int(rng.choice(tl)),
                    'dropout':float(rng.choice(tdr)),
                    'lr':float(rng.choice(lr))}})
        return out

    experiments = _gen_configs(N_EXPERIMENTS, MASTER_SEED)
    print(f"  Configs regenerated: {len(experiments)}")
    print()

    t0      = datetime.now()
    results = []

    # ── Checkpoint recovery — resume from last saved point ───────────────────
    checkpoint_path = os.path.join(RESULTS_DIR, 'eval_checkpoint_full.json')
    start_idx = 0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            saved = json.load(f)
        results   = saved
        start_idx = len(results)
        print(f"  Resuming from experiment {start_idx + 1} "
              f"({start_idx} already completed)\n")
    else:
        print(f"  Starting fresh — no checkpoint found\n")

    # ── Accumulators — rebuild from recovered results first ──────────────────
    all_val_residuals  = []
    all_val_within_std = []
    for r in results:
        all_val_residuals.extend(r['val_residuals'])
        all_val_within_std.extend(r['val_within_std'])

    for i, config in enumerate(experiments[start_idx:], start=start_idx):
        res = evaluate_experiment(config, i, N_EXPERIMENTS, arrays, INPUT_SHAPE)
        results.append(res)

        all_val_residuals.extend(res['val_residuals'])
        all_val_within_std.extend(res['val_within_std'])

        gc.collect()

        if (i + 1) % 10 == 0 or (i + 1) == N_EXPERIMENTS:
            # ── Full checkpoint (includes train_times, infer_times) ───────────
            with open(checkpoint_path, 'w') as f:
                json.dump(results, f)

            # ── Lightweight evaluation summary (strip heavy arrays) ───────────
            lightweight = [
                {k: v for k, v in r.items()
                 if k not in ('test_predictions', 'ensemble_test_pred',
                              'y_test', 'val_residuals', 'val_within_std',
                              'train_times', 'infer_times')}
                for r in results
            ]
            with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json'), 'w') as f:
                json.dump(lightweight, f, indent=2)

            # ── All predictions ───────────────────────────────────────────────
            pred_data = [{
                'experiment'        : r['experiment'],
                'seed'              : r['seed'],
                'ensemble_test_pred': r['ensemble_test_pred'],
                'test_predictions'  : r['test_predictions'],
                'y_test'            : r['y_test'],
            } for r in results]
            with open(os.path.join(RESULTS_DIR, 'all_predictions.json'), 'w') as f:
                json.dump(pred_data, f)

            elapsed = (datetime.now() - t0).total_seconds()
            rate    = elapsed / (i + 1 - start_idx)
            eta     = rate * (N_EXPERIMENTS - i - 1)
            print(f"\n  Progress: {i+1}/{N_EXPERIMENTS}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min\n")

    # ── Guard: only aggregate if we have results ──────────────────────────────
    if not results:
        print("  No results to aggregate — exiting.")
        return []

    # ── Aggregate computational cost ──────────────────────────────────────────
    all_train_times = {m: [r['train_times'][m] for r in results] for m in MODEL_NAMES}
    all_infer_times = {m: [r['infer_times'][m] for r in results] for m in MODEL_NAMES}

    compute_cost = {
        'training_time_seconds': {
            m: {
                'mean': float(np.mean(all_train_times[m])),
                'std' : float(np.std(all_train_times[m])),
                'min' : float(np.min(all_train_times[m])),
                'max' : float(np.max(all_train_times[m])),
            } for m in MODEL_NAMES
        },
        'inference_time_per_cycle_ms': {
            m: {
                'mean': float(np.mean(all_infer_times[m]) * 1000),
                'std' : float(np.std(all_infer_times[m])  * 1000),
            } for m in MODEL_NAMES
        },
        'total_training_time_seconds': {
            m: float(np.sum(all_train_times[m])) for m in MODEL_NAMES
        },
        'ensemble_inference_per_cycle_ms':
            float(sum(np.mean(all_infer_times[m]) for m in MODEL_NAMES) * 1000),
        'n_experiments': N_EXPERIMENTS,
        'note': (
            'Training time per model per experiment. '
            'Inference time = total test prediction / N_PRED cycles. '
            'Ensemble inference = sum of all 6 model inference times.'
        ),
    }

    with open(os.path.join(RESULTS_DIR, 'compute_cost.json'), 'w') as f:
        json.dump(compute_cost, f, indent=2)

    print(f"\n  ── Computational Cost Summary ──")
    for m in MODEL_NAMES:
        print(f"  {m:12s}  train={np.mean(all_train_times[m]):.1f}s  "
              f"infer={np.mean(all_infer_times[m])*1000:.3f}ms/cycle")
    print(f"  Ensemble infer/cycle : "
          f"{compute_cost['ensemble_inference_per_cycle_ms']:.3f} ms")
    print(f"  Saved: results/compute_cost.json")

    # ── Validation calibration data (unchanged from v3) ──────────────────────
    all_val_residuals  = np.array(all_val_residuals,  dtype=float)
    all_val_within_std = np.array(all_val_within_std, dtype=float)

    q90_val = float(np.percentile(all_val_residuals, 90))

    valid_mask = all_val_within_std > 0.1
    if valid_mask.sum() > 100:
        cal_ratios = all_val_residuals[valid_mask] / all_val_within_std[valid_mask]
        k90_val    = float(np.percentile(cal_ratios, 90))
    else:
        k90_val = float(np.percentile(all_val_residuals, 90) /
                        max(np.mean(all_val_within_std), 0.1))

    n_val_total         = len(all_val_residuals)
    delta               = 0.05
    conformal_tolerance = float(
        np.sqrt(np.log(2 / delta) / (2 * n_val_total)) * 100)

    print(f"\n  ── Validation Calibration Data ──")
    print(f"  Total val calibration cycles : {n_val_total}")
    print(f"  q90 (conformal half-width)   : {q90_val:.4f} cycles")
    print(f"  k90 (variance CI factor)     : {k90_val:.4f}")
    print(f"  Finite-sample tolerance (95%): ±{conformal_tolerance:.2f}%")
    print(f"  Valid std cycles for k90     : {valid_mask.sum()}/{n_val_total}")

    val_cal = {
        'q90'                        : q90_val,
        'k90'                        : k90_val,
        'n_val_cycles'               : n_val_total,
        'n_experiments'              : N_EXPERIMENTS,
        'finite_sample_tolerance_pct': conformal_tolerance,
        'valid_std_cycles'           : int(valid_mask.sum()),
        'val_residuals_mean'         : float(all_val_residuals.mean()),
        'val_residuals_std'          : float(all_val_residuals.std()),
        'val_residuals_p50'          : float(np.percentile(all_val_residuals, 50)),
        'val_residuals_p90'          : q90_val,
        'val_residuals_p95'          : float(np.percentile(all_val_residuals, 95)),
        'all_val_residuals'          : all_val_residuals.tolist(),
        'all_val_within_std'         : all_val_within_std.tolist(),
    }
    with open(os.path.join(RESULTS_DIR, 'val_calibration_data.json'), 'w') as f:
        json.dump(val_cal, f)
    print(f"  Saved: results/val_calibration_data.json")

    total_elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nAll {N_EXPERIMENTS} experiments complete in "
          f"{total_elapsed/60:.1f} minutes")

    ens_rmses = [r['ensemble_test']['rmse'] for r in results]
    print(f"Aggregate RMSE: mean={np.mean(ens_rmses):.2f}  "
          f"std={np.std(ens_rmses):.2f}  "
          f"best={np.min(ens_rmses):.2f}")

    return results


if __name__ == '__main__':
    run_all_evaluations()