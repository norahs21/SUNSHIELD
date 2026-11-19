"""Step 1 - load, merge and clean the Kaggle solar data; build the expected-power references.

Output tables
-------------
weather  : one row per plant and 15-minute timestamp (night included), the model input timeline
readings : one row per inverter and DAYLIGHT timestamp (IRRADIATION > 0) with
           AC_POWER, weather, SPLIT and the model-free expected power (EXPECTED_PEER_kW)
"""
import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- loading
def load_raw(raw_dir=C.RAW_DIR):
    """Read the 4 CSV files with their (different) date formats."""
    gen, wea = [], []
    for plant in C.PLANTS:
        for files, out in ((C.GEN_FILES, gen), (C.WEATHER_FILES, wea)):
            name = files[plant]
            df = pd.read_csv(raw_dir / name)
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format=C.DATE_FORMATS[name])
            out.append(df)
    return pd.concat(gen, ignore_index=True), pd.concat(wea, ignore_index=True)


def fill_short_gaps(series, max_steps=C.MAX_FILL_STEPS):
    """Linear interpolation of gaps of at most `max_steps` readings; longer gaps stay NaN."""
    is_na = series.isna()
    gap_id = (is_na != is_na.shift()).cumsum()
    gap_len = is_na.groupby(gap_id).transform('sum')
    filled = series.interpolate(method='linear', limit_area='inside')
    return series.where(~(is_na & (gap_len <= max_steps)), filled)


def weather_timeline(wea):
    """Full 15-minute grid per plant (night included), short gaps filled, time-of-day features."""
    parts = []
    for plant, w in wea.groupby('PLANT_ID'):
        start = w['DATE_TIME'].min().normalize()
        end = w['DATE_TIME'].max().normalize() + pd.Timedelta('1D') - pd.Timedelta(C.FREQ)
        grid = pd.date_range(start, end, freq=C.FREQ, name='DATE_TIME')
        g = w.set_index('DATE_TIME')[C.WEATHER].reindex(grid)
        g['MISSING_BEFORE_FILL'] = g['IRRADIATION'].isna()
        for col in C.WEATHER:
            g[col] = fill_short_gaps(g[col])
        g = g.reset_index()
        g.insert(0, 'PLANT_ID', plant)
        parts.append(g)
    weather = pd.concat(parts, ignore_index=True)
    minutes = weather['DATE_TIME'].dt.hour * 60 + weather['DATE_TIME'].dt.minute
    weather['TOD_SIN'] = np.sin(2 * np.pi * minutes / 1440)
    weather['TOD_COS'] = np.cos(2 * np.pi * minutes / 1440)
    return weather


def assign_split(dates, train_frac=C.TRAIN_FRAC, val_frac=C.VAL_FRAC):
    """Chronological split by whole days: 70 % train, 15 % validation, 15 % test."""
    days = np.sort(dates.dt.normalize().unique())
    n_train, n_val = round(len(days) * train_frac), round(len(days) * val_frac)
    labels = ['train'] * n_train + ['val'] * n_val + ['test'] * (len(days) - n_train - n_val)
    return dates.dt.normalize().map(pd.Series(labels, index=days))


# --------------------------------------------------------------------------- references
def leave_one_out_quantile(values, q=C.PEER_QUANTILE, min_peers=C.MIN_PEERS):
    """For every inverter: the q-quantile of the OTHER inverters at the same timestamp."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) - 1 < min_peers:
        return out
    for i in range(len(v)):
        out[i] = np.quantile(np.delete(v, i), q)
    return out


def physical_expectation(irradiation, module_temp, scale, gamma=C.GAMMA):
    """PVWatts-style model: P = scale * G * (1 + gamma * (T_module - 25))."""
    return scale * irradiation * (1 + gamma * (module_temp - 25.0))


def add_references(readings):
    """Model-free expected power of a HEALTHY inverter (used for the ground truth, never by the LSTM).

    1. Peer reference (Performance-to-Peers idea): the 75th percentile of the other inverters of the
       same plant at the same moment. They share the sky, so they show what a healthy inverter makes.
       The 75th percentile (not the median) stays correct even when up to 3/4 of the peers are down.
    2. Plant-wide events: when even the peers are far below what the irradiation allows
       (peers < 50 % of the physical expectation) there is no healthy neighbour, so the physical
       PV model is used instead.
    """
    r = readings.copy()
    r['PEER_P75_kW'] = r.groupby(['PLANT_ID', 'DATE_TIME'])[C.TARGET].transform(leave_one_out_quantile)

    # Scale of the physical model: fitted on TRAINING data only, sunny readings, healthy peers
    r['PHYS_UNIT'] = physical_expectation(r['IRRADIATION'], r['MODULE_TEMPERATURE'], 1.0)
    capacity, scale = {}, {}
    for plant, g in r[r['SPLIT'] == 'train'].groupby('PLANT_ID'):
        capacity[plant] = float(g[C.TARGET].quantile(0.999))
        sunny = g[(g['IRRADIATION'] > 0.2) & g['PEER_P75_kW'].notna()]
        scale[plant] = float(np.median(sunny['PEER_P75_kW'] / sunny['PHYS_UNIT']))
    r['CAPACITY_kW'] = r['PLANT_ID'].map(capacity)
    r['EXPECTED_PHYS_kW'] = r['PHYS_UNIT'] * r['PLANT_ID'].map(scale)
    r['FLOOR_kW'] = C.FLOOR_FRAC * r['CAPACITY_kW']

    plant_event = ((r['PEER_P75_kW'] < C.PLANT_EVENT_RATIO * r['EXPECTED_PHYS_kW'])
                   & (r['EXPECTED_PHYS_kW'] > r['FLOOR_kW']))
    r['PLANT_EVENT'] = plant_event
    r['EXPECTED_PEER_kW'] = np.where(plant_event, r['EXPECTED_PHYS_kW'], r['PEER_P75_kW'])
    return r.drop(columns='PHYS_UNIT'), {'capacity_kW': capacity, 'physics_scale': scale}


# --------------------------------------------------------------------------- full step 1
def build_dataset(raw_dir=C.RAW_DIR):
    gen, wea = load_raw(raw_dir)
    weather = weather_timeline(wea)
    readings = gen[['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER']].merge(
        weather[['PLANT_ID', 'DATE_TIME'] + C.WEATHER], on=['PLANT_ID', 'DATE_TIME'], how='left')
    n_merged = len(readings)
    readings = readings.dropna(subset=C.WEATHER)                       # weather gap > 1 h
    n_known_weather = len(readings)
    readings = readings[readings['IRRADIATION'] > 0].copy()            # night removed
    readings['SPLIT'] = assign_split(readings['DATE_TIME'])
    weather['SPLIT'] = assign_split(weather['DATE_TIME'])
    readings, params = add_references(readings)
    readings = readings.sort_values(['PLANT_ID', 'SOURCE_KEY', 'DATE_TIME']).reset_index(drop=True)
    counts = {'generation rows': len(gen), 'merged with weather': n_merged,
              'weather known': n_known_weather, 'daylight readings': len(readings)}
    return weather, readings, params, counts
