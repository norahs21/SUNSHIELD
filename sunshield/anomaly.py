"""Step 3 - the anomaly definition.

A reading is an anomaly when the inverter produced clearly MORE (rise, +1) or clearly LESS (drop, -1)
than a healthy inverter would under the same weather, for long enough not to be a passing cloud.

    d_t     = (actual_t - expected_t) / max(expected_t, floor)       normalised deviation (-1 = produced nothing)
    s_t     = EWMA of d over ~1 hour (span 4, past readings only)    smoothed deviation
    c_t     = median(s_0 ... s_t)                                    expanding centre, from 0
    sigma_t = max(1.4826 * median(|s_0..t - c_t|), SIGMA_MIN)        robust sigma, never below 10 %
    anomaly = +1 if s_t > c_t + k * sigma_t   (rise)
              -1 if s_t < c_t - k * sigma_t   (drop)
               0 otherwise

The SAME rule is applied to two lines, each alone:
    Actual 0/1    : expected = healthy peers (model-free, measured)  -> ground truth
    Predicted 0/1 : expected = LSTM normal-behaviour model           -> what the model detects
"""
import numpy as np
import pandas as pd

from . import config as C


def relative_deviation(actual, expected, floor):
    """(actual - expected) / max(expected, floor)."""
    actual, expected, floor = (np.asarray(v, dtype=float) for v in (actual, expected, floor))
    return (actual - expected) / np.maximum(expected, floor)


def ewma(x, span=C.EWMA_SPAN):
    """Exponentially weighted moving average that only uses the current and earlier readings."""
    if span is None or span <= 1:
        return np.asarray(x, dtype=float)
    return pd.Series(np.asarray(x, dtype=float)).ewm(span=span, adjust=False, ignore_na=True).mean().to_numpy()


def expanding_robust_stats(x, min_history=C.MIN_HISTORY):
    """centre_t = median and sigma_t = 1.4826 * MAD of x[0 .. t] (the reading and everything before it).

    NaN values are skipped. Before `min_history` readings exist, centre/sigma are NaN (no flag).
    """
    x = np.asarray(x, dtype=float)
    centre = np.full(len(x), np.nan)
    sigma = np.full(len(x), np.nan)
    history = []
    for t, v in enumerate(x):
        if np.isnan(v):
            continue
        history.append(v)
        if len(history) >= min_history:
            h = np.asarray(history)
            med = np.median(h)
            centre[t] = med
            sigma[t] = C.MAD_TO_SIGMA * np.median(np.abs(h - med))
    return centre, sigma


def expanding_mean_std(x, min_history=C.MIN_HISTORY):
    """Plain mean / std of x[0 .. t] (used only in the comparison, to show why the robust version is needed)."""
    s = pd.Series(np.asarray(x, dtype=float))
    n = s.notna().cumsum()
    mean = s.expanding().mean().where(n >= min_history)
    std = s.expanding().std(ddof=0).where(n >= min_history)
    return mean.to_numpy(), std.to_numpy()


def flags_from_stats(x, centre, sigma, k, sigma_min=C.SIGMA_MIN):
    """+1 rise / -1 drop / 0 normal, with sigma never below `sigma_min`. No history -> 0."""
    x = np.asarray(x, dtype=float)
    sigma = np.maximum(np.nan_to_num(sigma, nan=0.0), max(sigma_min, 1e-9))
    out = np.zeros(len(x), dtype=int)
    ok = ~np.isnan(x) & ~np.isnan(centre)
    out[ok & (x > centre + k * sigma)] = 1
    out[ok & (x < centre - k * sigma)] = -1
    return out


def flag_line(df, deviation_col, prefix, k_values=C.K_VALUES, span=C.EWMA_SPAN, sigma_min=C.SIGMA_MIN,
              robust=True):
    """Apply the anomaly definition to one deviation line, separately for every inverter, in time order.

    Adds `<prefix>_smooth`, `<prefix>_centre`, `<prefix>_sigma` (after the minimum) and one
    `<prefix>_k<k>` column per k with +1 / -1 / 0.
    """
    df = df.sort_values(['PLANT_ID', 'SOURCE_KEY', 'DATE_TIME']).copy()
    stats = expanding_robust_stats if robust else expanding_mean_std
    raw = df[deviation_col].to_numpy(dtype=float)
    smooth = np.full(len(df), np.nan)
    centre = np.full(len(df), np.nan)
    sigma = np.full(len(df), np.nan)
    for idx in df.groupby('SOURCE_KEY', sort=False).indices.values():
        smooth[idx] = ewma(raw[idx], span)
        centre[idx], sigma[idx] = stats(smooth[idx])
    sigma = np.where(np.isnan(sigma), np.nan, np.maximum(sigma, sigma_min))
    df[f'{prefix}_smooth'] = smooth
    df[f'{prefix}_centre'] = centre
    df[f'{prefix}_sigma'] = sigma
    for k in k_values:
        df[f'{prefix}_k{k:g}'] = flags_from_stats(smooth, centre, sigma, k, sigma_min)
    return df


# --------------------------------------------------------------------------- the previous definition
def previous_definition_flags(series, k, min_history=10):
    """The definition used before the rebuild (kept only for the comparison):
    change from the previous reading, outside mean +/- k*std of the PREVIOUS changes, on ONE line alone."""
    data = pd.Series(series).diff().fillna(0).to_numpy()
    flags = np.zeros(len(data), dtype=int)
    for t in range(min_history, len(data)):
        h = data[:t]
        m, s = np.mean(h), np.std(h)
        if s == 0:
            flags[t] = int(data[t] != m)
        elif data[t] > m + k * s or data[t] < m - k * s:
            flags[t] = 1
    return flags
