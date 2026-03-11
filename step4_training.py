# =============================================================================
# step4_training.py
# 100 Experiment Training Loop — 600 model training runs
# Trains all 6 models per experiment, computes WTA3 weights, saves results
# =============================================================================

import numpy as np
import json
import os
import gc
import warnings
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import median_abs_deviation
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Import shared objects from setup and model files
# Run step1_setup.py first to populate these variables
# -----------------------------------------------------------------------------

from step1_setup import (
    experiments, feature_cols, INPUT_SHAPE,
    X_train_flat, y_train_flat, X_val_flat, y_val_flat,
    X_train_seq, y_train_seq, X_val_seq, y_val_seq,
    RESULTS_DIR, N_EXPERIMENTS
)
from step3_models import build_model

MODEL_NAMES = ['RF', 'XGB', 'SVR', 'LSTM', 'CNN', 'Transformer']

# -----------------------------------------------------------------------------
# WTA3 weight calculation
# Inverse-power weighting on validation RMSE, with MAD-based outlier removal
# -----------------------------------------------------------------------------

def compute_wta3_weights(val_rmses, model_names, alpha=0.05, beta=-3):
    val_rmses = np.array(val_rmses)
    median_r  = np.median(val_rmses)
    mad       = median_abs_deviation(val_rmses)
    outlier   = np.abs(val_rmses - median_r) > 2.5 * mad

    mean_inlier = np.mean(val_rmses[~outlier]) if (~outlier).any() else np.mean(val_rmses)
    weights = np.zeros(len(val_rmses))
    for i, r in enumerate(val_rmses):
        if not outlier[i]:
            weights[i] = (r + alpha * mean_inlier) ** beta
    total = weights.sum()
    if total > 0:
        weights /= total

    return {n: float(w) for n, w in zip(model_names, weights)}, \
           {n: bool(o) for n, o in zip(model_names, outlier)}

# -----------------------------------------------------------------------------
# Single experiment training
# -----------------------------------------------------------------------------

def run_experiment(config, exp_idx):
    name = config['name']
    seed = config['seed']
    tf.random.set_seed(seed)
    np.random.seed(seed)

    t_start = datetime.now()
    print(f"  Experiment {exp_idx+1:03d}/{N_EXPERIMENTS}  [{name}]  seed={seed}")

    models      = {}
    train_rmses = {}
    val_rmses   = {}

    model_params = {
        'RF'         : config['rf_params'],
        'XGB'        : config['xgb_params'],
        'SVR'        : config['svr_params'],
        'LSTM'       : config['lstm_params'],
        'CNN'        : config['cnn_params'],
        'Transformer': config['transformer_params'],
    }

    for mname in MODEL_NAMES:
        params = model_params[mname]
        model  = build_model(mname, params, INPUT_SHAPE)

        if mname in ('RF', 'XGB', 'SVR'):
            if mname == 'XGB':
                model.fit(X_train_flat, y_train_flat,
                          eval_set=[(X_val_flat, y_val_flat)],
                          verbose=False)
            else:
                model.fit(X_train_flat, y_train_flat)
            tr_pred  = model.predict(X_train_flat)
            val_pred = model.predict(X_val_flat)
        else:
            cb = EarlyStopping(monitor='val_loss', patience=10,
                               min_delta=0.1, restore_best_weights=True,
                               verbose=0)
            model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50, batch_size=64,
                callbacks=[cb], verbose=0
            )
            tr_pred  = model.predict(X_train_seq, verbose=0).flatten()
            val_pred = model.predict(X_val_seq,   verbose=0).flatten()

        tr_rmse  = float(np.sqrt(mean_squared_error(y_train_flat if mname in ('RF','XGB','SVR') else y_train_seq, tr_pred)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val_flat   if mname in ('RF','XGB','SVR') else y_val_seq,   val_pred)))
        gap      = val_rmse / tr_rmse if tr_rmse > 0 else 1.0
        status   = 'GOOD' if gap < 1.5 else 'WARN' if gap < 2.0 else 'OVERFIT'

        print(f"    {mname:12s}  train={tr_rmse:6.2f}  val={val_rmse:6.2f}  gap={gap:.2f}x  {status}")

        models[mname]      = model
        train_rmses[mname] = tr_rmse
        val_rmses[mname]   = val_rmse

    # WTA3 weights
    rmse_list            = [val_rmses[m] for m in MODEL_NAMES]
    weights, outlier_map = compute_wta3_weights(rmse_list, MODEL_NAMES)

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"    WTA3 weights: " +
          "  ".join(f"{m}={weights[m]:.3f}" + (" [OUT]" if outlier_map[m] else "")
                    for m in MODEL_NAMES))
    print(f"    Time: {elapsed:.0f}s")
    print()

    return {
        'config'      : config['name'],
        'seed'        : seed,
        'models'      : models,
        'weights'     : weights,
        'outliers'    : outlier_map,
        'train_rmses' : train_rmses,
        'val_rmses'   : val_rmses,
        'elapsed_s'   : elapsed
    }

# -----------------------------------------------------------------------------
# Main training loop — all 100 experiments
# Results saved incrementally so a crash does not lose progress
# -----------------------------------------------------------------------------

def run_all_experiments():
    print("=" * 70)
    print("  STEP 4: Training 100 Experiments (600 model runs)")
    print("=" * 70)
    print()

    total_start    = datetime.now()
    all_results    = []
    summary_rows   = []

    for i, config in enumerate(experiments):
        result = run_experiment(config, i)
        all_results.append(result)

        # Save lightweight summary row (no model objects)
        summary_rows.append({
            'experiment'  : result['config'],
            'seed'        : result['seed'],
            'weights'     : result['weights'],
            'outliers'    : result['outliers'],
            'train_rmses' : result['train_rmses'],
            'val_rmses'   : result['val_rmses'],
            'elapsed_s'   : result['elapsed_s']
        })

        # Incremental save every 10 experiments
        if (i + 1) % 10 == 0 or (i + 1) == N_EXPERIMENTS:
            with open(os.path.join(RESULTS_DIR, 'training_summary.json'), 'w') as f:
                json.dump(summary_rows, f, indent=2)
            elapsed_total = (datetime.now() - total_start).total_seconds()
            rate          = elapsed_total / (i + 1)
            remaining     = rate * (N_EXPERIMENTS - i - 1)
            print(f"  Progress: {i+1}/{N_EXPERIMENTS} experiments complete  "
                  f"elapsed={elapsed_total/60:.1f}min  "
                  f"ETA={remaining/60:.1f}min")
            print()

        gc.collect()

    total_elapsed = (datetime.now() - total_start).total_seconds()
    print(f"All {N_EXPERIMENTS} experiments complete in {total_elapsed/60:.1f} minutes")
    print(f"Summary saved to results/training_summary.json")
    print()
    print("=" * 70)
    print("  STEP 4 COMPLETE")
    print("=" * 70)

    return all_results

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    all_results = run_all_experiments()