"""All settings of the SUNSHIELD pipeline in one place."""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
OUT_DIR = PROJECT_ROOT / 'outputs'          # created by the notebooks (git-ignored)
RESULTS_DIR = PROJECT_ROOT / 'results'      # small final tables and figures (committed)

GEN_FILES = {4135001: 'Plant_1_Generation_Data.csv', 4136001: 'Plant_2_Generation_Data.csv'}
WEATHER_FILES = {4135001: 'Plant_1_Weather_Sensor_Data.csv', 4136001: 'Plant_2_Weather_Sensor_Data.csv'}
DATE_FORMATS = {
    'Plant_1_Generation_Data.csv': '%d-%m-%Y %H:%M',
    'Plant_1_Weather_Sensor_Data.csv': '%Y-%m-%d %H:%M:%S',
    'Plant_2_Generation_Data.csv': '%Y-%m-%d %H:%M:%S',
    'Plant_2_Weather_Sensor_Data.csv': '%Y-%m-%d %H:%M:%S',
}
PLANTS = [4135001, 4136001]

# ----------------------------------------------------------------------------- preprocessing
FREQ = '15min'
MAX_FILL_STEPS = 4                  # weather gaps of at most 1 hour are interpolated
WEATHER = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
TIME_FEATURES = ['TOD_SIN', 'TOD_COS']   # time of day on a circle (sun position)
FEATURES = WEATHER + TIME_FEATURES
TARGET = 'AC_POWER'
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test = the remaining 15 %, split by whole days
SPLITS = ['train', 'val', 'test']
SPLIT_NAMES = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}

# ----------------------------------------------------------------------------- expected power (references)
PEER_QUANTILE = 0.75     # healthy-peer reference = 75th percentile of the OTHER inverters at the same time
MIN_PEERS = 5            # need at least 5 other inverters reporting
GAMMA = -0.004           # PV power temperature coefficient (1/degC), typical crystalline silicon
PLANT_EVENT_RATIO = 0.5  # peers below 50 % of the physical expectation = whole plant down -> use physics
FLOOR_FRAC = 0.20        # deviation is divided by max(expected, 20 % of capacity): relative above ~200 W/m2,
                         # absolute (in units of 20 % of capacity) at dawn, dusk and under heavy cloud

# ----------------------------------------------------------------------------- LSTM (normal behaviour model)
SEED = 42
WINDOW_SIZES = [4, 8, 16, 32, 48]  # 1 h ... 12 h of weather history
EPOCHS = 400
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 30
N_SEEDS = 3              # small ensemble: average of 3 networks trained with different seeds
CLEAN_Z = 3.0            # training targets: drop readings > 3 sigma away from the healthy peers

# ----------------------------------------------------------------------------- anomaly definition
K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
EWMA_SPAN = 4            # deviation smoothed with an exponential moving average over ~1 hour (4 readings)
MIN_HISTORY = 20         # a reading needs at least 20 earlier readings of its inverter before it can be flagged
MAD_TO_SIGMA = 1.4826    # sigma = 1.4826 * MAD (consistent with the std for normally distributed data)
SIGMA_MIN = 0.10         # sigma is never below 10 % of the expected power (measurement / spatial tolerance)
