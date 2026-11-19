"""Step 2 - the LSTM normal-behaviour model (NBM).

The LSTM learns the power of a HEALTHY inverter from the weather history (irradiation, ambient and
module temperature, time of day). It is trained on healthy readings only, so when an inverter
fails its actual power moves away from the LSTM line and the residual shows the fault.
"""
import os

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- inputs
def fit_feature_scaler(weather):
    """Mean / std of the weather features, from the TRAINING days only."""
    train = weather[weather['SPLIT'] == 'train']
    stats = {}
    for plant, g in train.groupby('PLANT_ID'):
        stats[plant] = {c: (float(g[c].mean()), float(g[c].std())) for c in C.WEATHER}
    return stats


def scaled_timeline(weather, stats):
    """Weather timeline with scaled features, per plant."""
    out = {}
    for plant, g in weather.groupby('PLANT_ID'):
        g = g.sort_values('DATE_TIME').reset_index(drop=True)
        X = g[C.FEATURES].to_numpy(np.float32).copy()
        for j, c in enumerate(C.WEATHER):
            mu, sd = stats[plant][c]
            X[:, j] = (X[:, j] - mu) / sd
        out[plant] = (g['DATE_TIME'].to_numpy(), X)
    return out


def build_windows(timeline, plant, timestamps, window):
    """X[i] = the `window` weather readings that END at timestamps[i] (the moment being predicted).

    The window runs on the continuous 15-minute weather timeline (night included), so every daylight
    reading gets a window and no window jumps over a night. A window with a weather gap > 1 h is invalid.
    """
    times, feats = timeline[plant]
    pos = pd.Index(times).get_indexer(pd.to_datetime(timestamps))
    valid = (pos >= window - 1)
    idx = pos[valid][:, None] + np.arange(-window + 1, 1)[None, :]
    X = feats[idx]
    ok = ~np.isnan(X).any(axis=(1, 2))
    valid[np.flatnonzero(valid)[~ok]] = False
    return X[ok], valid


# --------------------------------------------------------------------------- training targets
def healthy_targets(readings, z=C.CLEAN_Z):
    """Per plant and timestamp: mean AC power of the HEALTHY inverters (+ how many there were).

    Healthy = the reading's deviation from the healthy peers is within `z` sigma of that inverter's usual
    value (centre = median, sigma = max(1.4826 * MAD, SIGMA_MIN), from the training days only).
    Faulty readings are left out, as done for normal-behaviour models, so the LSTM never learns that
    "0 kW in full sun" is normal.
    """
    from .anomaly import relative_deviation
    r = readings.copy()
    r['DEV'] = relative_deviation(r[C.TARGET], r['EXPECTED_PEER_kW'], r['FLOOR_kW'])
    tr = r[r['SPLIT'] == 'train'].groupby('SOURCE_KEY')['DEV']
    centre = tr.median()
    sigma = (C.MAD_TO_SIGMA * tr.apply(lambda s: np.median(np.abs(s - s.median())))).clip(lower=C.SIGMA_MIN)
    zscore = (r['DEV'] - r['SOURCE_KEY'].map(centre)) / r['SOURCE_KEY'].map(sigma)
    r['HEALTHY'] = (zscore.abs() <= z) & ~r['PLANT_EVENT']
    t = (r[r['HEALTHY']].groupby(['PLANT_ID', 'DATE_TIME', 'SPLIT'])[C.TARGET]
         .agg(TARGET_kW='mean', N_HEALTHY='size').reset_index())
    return t, r['HEALTHY']


# --------------------------------------------------------------------------- the network
def build_lstm(window, n_features=len(C.FEATURES)):
    import keras
    from keras import layers
    model = keras.Sequential([
        keras.Input(shape=(window, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='softplus'),     # power can never be negative
    ], name=f'lstm_nbm_w{window}')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=C.LEARNING_RATE), loss='mse')
    return model


def plant_data(timeline, targets, plant, window, capacity):
    """Windows and scaled targets (power / capacity) for one plant, per split."""
    data = {}
    t = targets[targets['PLANT_ID'] == plant]
    for s in C.SPLITS:
        ts = t[t['SPLIT'] == s]
        X, valid = build_windows(timeline, plant, ts['DATE_TIME'], window)
        ts = ts[valid]
        data[s] = (X, (ts['TARGET_kW'] / capacity).to_numpy(np.float32),
                   ts['N_HEALTHY'].to_numpy(np.float32))
    return data


def train_one(data, window, seed, log_path=None, verbose=0):
    import keras
    keras.utils.set_random_seed(seed)
    model = build_lstm(window)
    X_tr, y_tr, w_tr = data['train']
    X_va, y_va, w_va = data['val']
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=C.PATIENCE,
                                               restore_best_weights=True)]
    if log_path:
        callbacks.append(keras.callbacks.CSVLogger(str(log_path)))
    hist = model.fit(X_tr, y_tr, sample_weight=w_tr, validation_data=(X_va, y_va, w_va),
                     epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, shuffle=True,
                     callbacks=callbacks, verbose=verbose).history
    return model, hist


def predict_readings(models, timeline, readings, window, capacity):
    """LSTM expected power (kW) for every reading; averaged over the ensemble of seeds."""
    out = pd.Series(np.nan, index=readings.index)
    for plant, g in readings.groupby('PLANT_ID'):
        keys = g[['DATE_TIME']].drop_duplicates()
        X, valid = build_windows(timeline, plant, keys['DATE_TIME'], window)
        preds = np.mean([m.predict(X, batch_size=4096, verbose=0).ravel() for m in models[plant]], axis=0)
        keys = keys[valid].assign(PRED=preds * capacity[plant])
        out.loc[g.index] = g[['DATE_TIME']].merge(keys, on='DATE_TIME', how='left')['PRED'].to_numpy()
    return out


def set_quiet_tf():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
