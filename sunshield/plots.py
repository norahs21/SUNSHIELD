"""Chart style and the figures shared by the notebooks (light surface, recessive grid, 2 px lines)."""
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'      # categorical slots 1-3
CRITICAL = '#d03b3b'                                       # status colour: anomaly
BAND = '#cde2fb'                                           # light blue: the normal band
CM_CMAP = 'Blues'

ACTUAL_C, LSTM_C, PEER_C = BLUE, ORANGE, AQUA


def use_style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
        'axes.titlecolor': INK, 'axes.titlelocation': 'left', 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 2, 'legend.frameon': False, 'font.size': 10})


def plot_confusion(ax, cm, title):
    """2x2 confusion matrix (rows = Actual 0/1, columns = Predicted 0/1)."""
    ax.imshow(cm, cmap=CM_CMAP)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else INK, fontsize=11)
    ax.set_xticks([0, 1], ['Normal (0)', 'Anomaly (1)'])
    ax.set_yticks([0, 1], ['Normal (0)', 'Anomaly (1)'])
    ax.set_xlabel('Predicted (LSTM line)')
    ax.set_ylabel('Actual (ground truth)')
    ax.set_title(title, fontsize=10)


def plot_inverter_day(ax, g, title, flag_col=None, flag_label='Anomaly', show_peer=True, show_lstm=True):
    """Actual power of one inverter with the expected lines; anomalies marked with red dots."""
    t = g['DATE_TIME']
    ax.plot(t, g['AC_POWER'], color=ACTUAL_C, label='Actual (this inverter)')
    if show_peer and 'EXPECTED_PEER_kW' in g:
        ax.plot(t, g['EXPECTED_PEER_kW'], color=PEER_C, linestyle='--', label='Healthy peers (p75)')
    if show_lstm and 'EXPECTED_LSTM_kW' in g:
        ax.plot(t, g['EXPECTED_LSTM_kW'], color=LSTM_C, label='LSTM expected')
    if flag_col is not None:
        f = g[g[flag_col] != 0]
        ax.plot(f['DATE_TIME'], f['AC_POWER'], 'o', color=CRITICAL, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=flag_label, linestyle='none')
    ax.set_ylabel('AC power (kW)')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper left', fontsize=8)


def plot_deviation_band(ax, g, raw_col, smooth_col, centre_col, sigma_col, flag_col, k, title):
    """The deviation line with its expanding centre +/- k*sigma band (the sketch of the definition)."""
    t = g['DATE_TIME']
    lo = g[centre_col] - k * g[sigma_col]
    hi = g[centre_col] + k * g[sigma_col]
    ax.fill_between(t, lo, hi, color=BAND, linewidth=0, label=f'centre ± {k:g}·σ (from 0 … t)')
    ax.plot(t, g[raw_col], color=GRID, linewidth=1.2, label='deviation (raw)')
    ax.plot(t, g[smooth_col], color=INK_2, linewidth=1.8, label='deviation (1-hour EWMA)')
    f = g[g[flag_col] != 0]
    ax.plot(f['DATE_TIME'], f[smooth_col], 'o', color=CRITICAL, markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.2, linestyle='none', label='anomaly (rise / drop)')
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_ylabel('normalised deviation')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='lower left', fontsize=8)


def plot_metrics_vs_k(ax, table, split_name, best_k=None):
    m = table[table['Split'] == split_name]
    ax.plot(m['k'], m['Precision'], marker='o', markersize=8, color=BLUE, label='Precision')
    ax.plot(m['k'], m['Recall'], marker='s', markersize=8, color=ORANGE, label='Recall')
    ax.plot(m['k'], m['F1'], marker='^', markersize=8, color=AQUA, label='F1')
    if best_k is not None:
        ax.axvline(best_k, color=INK_2, linestyle='--', linewidth=1)
        ax.annotate(f'best k = {best_k:g}', (best_k, 0.05), textcoords='offset points', xytext=(6, 0),
                    color=INK)
    ax.set_xticks(np.unique(m['k']))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('k (number of σ)')
    ax.set_title(split_name, fontsize=10)
