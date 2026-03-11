# =============================================================================
# step3_models.py
# Model Building Functions — RF, XGB, SVR, LSTM, CNN, Transformer
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# -----------------------------------------------------------------------------
# Traditional ML
# -----------------------------------------------------------------------------

def build_rf(params):
    return RandomForestRegressor(**params)

def build_xgb(params):
    return xgb.XGBRegressor(**params, tree_method='hist', verbosity=0)

def build_svr(params):
    return SVR(**params)

# -----------------------------------------------------------------------------
# LSTM
# -----------------------------------------------------------------------------

def build_lstm(params, input_shape):
    model = keras.Sequential([
        layers.LSTM(
            params['units'],
            return_sequences=False,
            dropout=params['dropout'],
            recurrent_dropout=params['recurrent_dropout'],
            kernel_regularizer=regularizers.l2(params['l2_reg']),
            input_shape=input_shape
        ),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(params['l2_reg'])),
        layers.Dropout(params['dropout']),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(params['lr']),
        loss='mse', metrics=['mae']
    )
    return model

# -----------------------------------------------------------------------------
# CNN
# -----------------------------------------------------------------------------

def build_cnn(params, input_shape):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    for _ in range(params['layers']):
        model.add(layers.Conv1D(
            params['filters'], params['kernel_size'],
            activation='relu', padding='same',
            kernel_regularizer=regularizers.l2(params['l2_reg'])
        ))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(params['dropout']))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(params['l2_reg'])))
    model.add(layers.Dropout(params['dropout']))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(params['lr']),
        loss='mse', metrics=['mae']
    )
    return model

# -----------------------------------------------------------------------------
# Transformer
# -----------------------------------------------------------------------------

def build_transformer(params, input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(params['d_model'])(inputs)

    for _ in range(params['num_layers']):
        attn = layers.MultiHeadAttention(
            num_heads=params['num_heads'],
            key_dim=max(1, params['d_model'] // params['num_heads']),
            dropout=params['dropout']
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ffn_out = keras.Sequential([
            layers.Dense(params['d_model'] * 2, activation='relu'),
            layers.Dropout(params['dropout']),
            layers.Dense(params['d_model'])
        ])(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(params['dropout'])(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(params['lr']),
        loss='mse', metrics=['mae']
    )
    return model

# -----------------------------------------------------------------------------
# Unified builder — called by training loop with model name + params
# -----------------------------------------------------------------------------

def build_model(name, params, input_shape):
    if name == 'RF':
        return build_rf(params)
    elif name == 'XGB':
        return build_xgb(params)
    elif name == 'SVR':
        return build_svr(params)
    elif name == 'LSTM':
        return build_lstm(params, input_shape)
    elif name == 'CNN':
        return build_cnn(params, input_shape)
    elif name == 'Transformer':
        return build_transformer(params, input_shape)
    else:
        raise ValueError(f"Unknown model: {name}")

# -----------------------------------------------------------------------------
# Verify all 6 builds correctly with dummy data
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("Testing model builds with dummy data...")

    dummy_shape  = (50, 22)
    dummy_seq    = np.random.randn(10, 50, 22).astype(np.float32)
    dummy_flat   = np.random.randn(10, 22).astype(np.float32)
    dummy_target = np.random.randn(10).astype(np.float32)

    rf_p   = dict(n_estimators=10, max_depth=3, min_samples_split=2,
                  min_samples_leaf=1, max_features=0.5, random_state=42)
    xgb_p  = dict(learning_rate=0.1, max_depth=3, n_estimators=10,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0, min_child_weight=1,
                  random_state=42)
    svr_p  = dict(C=10, gamma='scale', epsilon=0.5)
    lstm_p = dict(units=16, layers=1, dropout=0.1, recurrent_dropout=0.1,
                  lr=0.001, l2_reg=0.001)
    cnn_p  = dict(filters=16, kernel_size=3, layers=1, dropout=0.1,
                  lr=0.001, l2_reg=0.001)
    trans_p = dict(num_heads=2, d_model=16, num_layers=1,
                   dropout=0.1, lr=0.001)

    test_configs = [
        ('RF',          rf_p,    dummy_flat, dummy_flat),
        ('XGB',         xgb_p,   dummy_flat, dummy_flat),
        ('SVR',         svr_p,   dummy_flat, dummy_flat),
        ('LSTM',        lstm_p,  dummy_seq,  dummy_seq),
        ('CNN',         cnn_p,   dummy_seq,  dummy_seq),
        ('Transformer', trans_p, dummy_seq,  dummy_seq),
    ]

    all_ok = True
    for name, params, X_tr, X_val in test_configs:
        try:
            m = build_model(name, params, dummy_shape)
            if name in ('RF', 'XGB', 'SVR'):
                m.fit(X_tr, dummy_target)
                pred = m.predict(X_val)
            else:
                m.fit(X_tr, dummy_target, epochs=1, batch_size=5, verbose=0)
                pred = m.predict(X_val, verbose=0).flatten()
            print(f"  {name:12s}  build+predict OK  output shape: {pred.shape}")
        except Exception as e:
            print(f"  {name:12s}  FAILED: {e}")
            all_ok = False

    print()
    print("All models verified." if all_ok else "Some models failed — check errors above.")