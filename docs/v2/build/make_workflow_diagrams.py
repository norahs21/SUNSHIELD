# Rebuild: python docs/v2/build/make_workflow_diagrams.py outputs docs/v2/SUNSHIELD_v2_workflow_diagrams.pdf docs/v2/diagrams  (run the 5 notebooks first)
"""SUNSHIELD v2 — workflow diagrams (PDF + PNG pages). Reads the final numbers from the notebook outputs.
usage: python diagrams.py <outputs_dir> <out_pdf> <png_dir>"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages

OUT, PDF, PNG = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(PNG, exist_ok=True)
S4 = os.path.join(OUT, 'step4_outputs')
resA = pd.read_csv(os.path.join(S4, 'A_plain', 'results_best_k.csv')).set_index('Split')
allk = pd.read_csv(os.path.join(S4, 'A_plain', 'metrics_all_k.csv'))
exp1 = pd.read_csv(os.path.join(S4, 'experiments', 'exp1_training_data_best_k.csv'))
exp2 = pd.read_csv(os.path.join(S4, 'experiments', 'exp2_previous_rules.csv'))
outc = pd.read_csv(os.path.join(S4, 'experiments', 'outage_check.csv'))
BEST_K = float(resA['k'].iloc[0])
_sh = pd.read_csv(os.path.join(OUT, 'step2_outputs', 'healthy_share.csv')).set_index(['PLANT_ID', 'SPLIT'])
P2_FAULTY = 1 - _sh.loc[(4136001, 'train'), 'healthy share']
F = resA['F1']

# ---------------------------------------------------------------- style
INK, INK2, MUTED = '#1f1e1c', '#4a4844', '#7a7770'
STY = {
    'common':  dict(fc='#efede9', ec='#8d8a83', tx=INK),
    'actual':  dict(fc='#dcf0e5', ec='#1f7a52', tx='#0f4d33'),
    'lstm':    dict(fc='#e5e2f8', ec='#4b3fa3', tx='#2c2370'),
    'anomaly': dict(fc='#fbe2d5', ec='#b4471d', tx='#7a2a0c'),
    'phys':    dict(fc='#ddeef7', ec='#2a6f97', tx='#123f5a'),
    'result':  dict(fc='#fff6d6', ec='#b08900', tx='#5c4700'),
}
NEW_C = '#d9480f'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

def page(title, subtitle=None, size=(8.27, 11.69)):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100 * size[1] / size[0]); ax.axis('off')
    top = 100 * size[1] / size[0]
    ax.text(5, top - 5, title, fontsize=15, weight='bold', color=INK, va='top')
    if subtitle:
        ax.text(5, top - 9.2, subtitle, fontsize=9.5, color=INK2, va='top')
    return fig, ax, top

def box(ax, cx, cy, w, h, title, sub=None, kind='common', new=False, fs=10.5, sfs=8):
    st = STY[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=1.4',
                                fc=st['fc'], ec=st['ec'], lw=1.4, zorder=2))
    if sub:
        ax.text(cx, cy + h * 0.2, title, ha='center', va='center', fontsize=fs, weight='bold', color=st['tx'], zorder=3)
        ax.text(cx, cy - h * 0.17, sub, ha='center', va='center', fontsize=sfs, color=INK2, zorder=3, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha='center', va='center', fontsize=fs, weight='bold', color=st['tx'], zorder=3)
    if new:
        ax.text(cx + w / 2 - 1, cy + h / 2, ' NEW ' if new is True else f' {new} ', ha='right', va='center', fontsize=7,
                weight='bold', color='white', zorder=4,
                bbox=dict(boxstyle='round,pad=0.25', fc=NEW_C, ec=NEW_C))
    return dict(cx=cx, cy=cy, w=w, h=h, top=(cx, cy + h / 2), bot=(cx, cy - h / 2),
                left=(cx - w / 2, cy), right=(cx + w / 2, cy))

def line(ax, pts, arrow=True, color='#5b5853', lw=1.4, ls='-'):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls, zorder=1, solid_capstyle='round')
    if arrow:
        ax.annotate('', xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                                                       mutation_scale=11, shrinkA=0, shrinkB=0), zorder=1)

def legend(ax, y, items):
    x = 5
    for kind, label in items:
        st = STY[kind]
        ax.add_patch(FancyBboxPatch((x, y - 1.1), 3.2, 2.2, boxstyle='round,pad=0,rounding_size=0.5', fc=st['fc'], ec=st['ec'], lw=1.1))
        ax.text(x + 4.2, y, label, va='center', fontsize=7.5, color=INK2)
        x += 4.2 + len(label) * 0.95 + 2.2
    ax.text(x, y, ' NEW ', va='center', fontsize=7, weight='bold', color='white', bbox=dict(boxstyle='round,pad=0.25', fc=NEW_C, ec=NEW_C))
    ax.text(x + 6, y, 'new / changed', va='center', fontsize=7.5, color=INK2)

pages = []

# ================================================================ page 1: training
fig, ax, top = page('1. Training phase — from dataset to best σ',
                    "Dr. Mai's workflow, same order of boxes. Orange badges mark what is new in v2.")
y = top - 17
b_data = box(ax, 50, y, 40, 7, 'Dataset', 'Kaggle Solar Power Generation: 2 plants, 44 inverters,\n15-minute readings, 15 May – 17 June 2020')
b_pre = box(ax, 50, y - 12, 52, 7.5, 'Preprocessing  (unchanged)', 'merge generation + weather · fill weather gaps ≤ 1 h · remove night\n(irradiation = 0) · StandardScaler fitted on training only')
b_split = box(ax, 50, y - 24, 40, 6.5, 'Data split  (unchanged)', '70 % / 15 % / 15 %  by whole days\ntrain 15 May–7 Jun · val 8–12 Jun · test 13–17 Jun')
b_phys = box(ax, 50, y - 36, 56, 7.5, 'Expected power from physics (PVWatts)', 'P_exp = P_ref · G · (1 + γ (T_module − 25 °C)),  γ = −0.47 %/°C\nP_ref per inverter = median on training days (G ≥ 0.2 kW/m²)', kind='phys', new=True)
b_h = box(ax, 50, y - 48, 56, 7.5, 'Healthy readings', 'r = actual − P_exp;  healthy if  |r − median| ≤ 3σ  (σ = 1.4826·MAD,\ntraining days) → faulty readings left out of the LSTM targets', kind='phys', new=True)
b_act = box(ax, 10, y - 61, 16, 10, 'Actual data', 'training split\n+ PVWatts\nexpected', kind='actual', sfs=7.5)
b_lstm = box(ax, 50, y - 61, 54, 8.5, 'Training (LSTM)', 'window 48 (12 h) of weather · 64 → 32 units (unchanged)\ntrained on HEALTHY readings only → gives predicted line', kind='lstm', new='CHANGED')
b_def = box(ax, 51, y - 75, 64, 9.5, 'Anomaly definition using different σ', 'k = 1, 1.5, 2, 2.5, 3 · each line alone:  r = actual − expected\n1 if r < μ − kσ or r > μ + kσ   (μ = median, σ = 1.4826·MAD of r, per inverter, training days)\nreadings with irradiation < 0.2 kW/m² are not evaluated', kind='anomaly', new='CHANGED', sfs=7.8)
b_a01 = box(ax, 30, y - 89, 30, 7.5, 'Actual 0/1', 'expected = PVWatts line\none column per σ', kind='actual')
b_p01 = box(ax, 70, y - 89, 30, 7.5, 'Predicted 0/1', 'expected = LSTM line\none column per σ', kind='lstm')
b_f1 = box(ax, 50, y - 101, 38, 7, 'F1 for each σ', '+ confusion matrix (training split)')
tr = allk[allk['Split'] == 'Training'].set_index('k')['F1']
b_best = box(ax, 50, y - 112.5, 62, 8, f'Best σ:  k = {BEST_K:g}  (highest training F1 = {tr.max():.3f})',
             'training F1:  ' + '  ·  '.join(f'k={k:g}: {f:.3f}' for k, f in tr.items()), kind='anomaly', fs=10, sfs=7.8)
for a, b in [(b_data, b_pre), (b_pre, b_split), (b_split, b_phys), (b_phys, b_h)]:
    line(ax, [a['bot'], b['top']])
line(ax, [b_h['bot'], b_lstm['top']])
line(ax, [b_lstm['bot'], (50, b_def['top'][1])])
line(ax, [b_act['bot'], (b_act['cx'], b_def['cy']), b_def['left']])
line(ax, [(30, b_def['bot'][1]), b_a01['top']]); line(ax, [(70, b_def['bot'][1]), b_p01['top']])
line(ax, [b_a01['bot'], (30, b_f1['cy']), b_f1['left']]); line(ax, [b_p01['bot'], (70, b_f1['cy']), b_f1['right']])
line(ax, [b_f1['bot'], b_best['top']])
legend(ax, 5, [('common', 'common step'), ('phys', 'physics'), ('actual', 'actual line'), ('lstm', 'LSTM line'), ('anomaly', 'anomaly rule')])
pages.append(('1_training_phase', fig))

# ================================================================ page 2: testing
fig, ax, top = page('2. Testing phase — final result',
                    'Nothing is re-fitted on the test days: the LSTM, P_ref, μ, σ and k all come from training.')
y = top - 18
b_t = box(ax, 50, y, 36, 6.5, 'Test data', '13 – 17 June 2020 (15 %)')
b_act = box(ax, 10, y - 14, 16, 10, 'Actual data', 'test split\n+ PVWatts\nexpected', kind='actual', sfs=7.5)
b_lstm = box(ax, 50, y - 14, 50, 8.5, 'LSTM (trained)', 'the healthy-trained model from training\ngives the predicted line for the test days', kind='lstm')
b_def = box(ax, 51, y - 29, 64, 9, 'Anomaly definition', f'using the best σ from training only: k = {BEST_K:g}\nμ and σ of every inverter from the training days (same limits as training)\nonly readings with irradiation ≥ 0.2 kW/m² are evaluated', kind='anomaly')
b_a01 = box(ax, 30, y - 43, 30, 6.5, 'Actual 0/1', 'actual vs PVWatts', kind='actual')
b_p01 = box(ax, 70, y - 43, 30, 6.5, 'Predicted 0/1', 'actual vs LSTM', kind='lstm')
b_f1 = box(ax, 50, y - 56, 34, 7.5, 'F1 + confusion matrix', f'final test result:  F1 = {F["Testing"]:.3f}\n(precision {resA.loc["Testing", "Precision"]:.3f}, recall {resA.loc["Testing", "Recall"]:.3f})', kind='result')
line(ax, [b_t['bot'], (50, b_lstm['top'][1])])
line(ax, [b_lstm['bot'], (50, b_def['top'][1])])
line(ax, [b_act['bot'], (b_act['cx'], b_def['cy']), b_def['left']])
line(ax, [(30, b_def['bot'][1]), b_a01['top']]); line(ax, [(70, b_def['bot'][1]), b_p01['top']])
line(ax, [b_a01['bot'], (30, b_f1['cy']), b_f1['left']]); line(ax, [b_p01['bot'], (70, b_f1['cy']), b_f1['right']])
# results table
ty = y - 70
ax.text(5, ty, f'Results with k = {BEST_K:g} (Version A, plain 0/1)', fontsize=11, weight='bold', color=INK)
cols = ['Split', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1']
xs = [8, 27, 37, 47, 57, 70, 82, 93]
for x, c in zip(xs, cols):
    ax.text(x, ty - 5, c, fontsize=9, weight='bold', color=INK2, ha='left' if c == 'Split' else 'center')
for i, s in enumerate(['Training', 'Validation', 'Testing']):
    r = resA.loc[s]; yy = ty - 10 - i * 4.5
    if s == 'Testing':
        ax.add_patch(FancyBboxPatch((6, yy - 2), 90, 4, boxstyle='round,pad=0,rounding_size=0.8', fc='#fff6d6', ec='none'))
    vals = [s, f'{int(r.TP):,}', f'{int(r.FP):,}', f'{int(r.FN):,}', f'{int(r.TN):,}', f'{r.Precision:.3f}', f'{r.Recall:.3f}', f'{r.F1:.3f}']
    for x, v, c in zip(xs, vals, cols):
        ax.text(x, yy, v, fontsize=9, color=INK, ha='left' if c == 'Split' else 'center', va='center',
                weight='bold' if c == 'F1' else 'normal')
o = outc[(outc['Split'] == 'Testing')].set_index('Line')
n_out = int(o['peer-confirmed outage readings'].iloc[0])
ax.text(5, ty - 30, 'Independent check (does not use PVWatts or k):', fontsize=9.5, weight='bold', color=INK)
ax.text(5, ty - 33, f'Test readings where an inverter gives ≤ 5 % of its neighbours\' median power (neighbours > 100 kW): {n_out}.\n'
        f'Flagged by the healthy-trained LSTM: {int(o.loc["LSTM healthy only (main)", "flagged"])}  ·  '
        f'by the LSTM trained on all readings: {int(o.loc["LSTM all readings", "flagged"])}.', fontsize=9, color=INK2, va='top', linespacing=1.5)
legend(ax, 5, [('common', 'common step'), ('actual', 'actual line'), ('lstm', 'LSTM line'), ('anomaly', 'anomaly rule'), ('result', 'result')])
pages.append(('2_testing_phase', fig))

# ================================================================ page 3: inside the anomaly definition
fig, ax, top = page('3. Inside the anomaly definition — one reading',
                    'The same rule is applied to each line alone; only the "expected power" differs.')
y = top - 18
b_r = box(ax, 50, y, 56, 7, 'One reading of one inverter at time t', 'actual AC power  A_t  (kW)')
b_e1 = box(ax, 25, y - 14, 40, 9, 'Expected (actual line)', 'PVWatts:  P_ref · G_t · (1 + γ (T_t − 25))\nfrom irradiation + module temperature', kind='phys')
b_e2 = box(ax, 75, y - 14, 40, 9, 'Expected (predicted line)', 'LSTM:  last 12 h of weather → power\ntrained on healthy readings', kind='lstm')
b_r1 = box(ax, 25, y - 29, 40, 8, 'r_actual = A_t − PVWatts_t', 'deviation from physics', kind='actual')
b_r2 = box(ax, 75, y - 29, 40, 8, 'r_pred = A_t − LSTM_t', 'deviation from the model', kind='lstm')
b_l1 = box(ax, 25, y - 44, 40, 9.5, 'Limits of this inverter', 'μ = median of r_actual (training days)\nσ = 1.4826 · MAD of r_actual\n1 if r < μ − kσ or r > μ + kσ', kind='anomaly')
b_l2 = box(ax, 75, y - 44, 40, 9.5, 'Limits of this inverter', 'μ = median of r_pred (training days)\nσ = 1.4826 · MAD of r_pred\n1 if r < μ − kσ or r > μ + kσ', kind='anomaly')
b_o1 = box(ax, 25, y - 58, 26, 6, 'Actual 0/1', kind='actual')
b_o2 = box(ax, 75, y - 58, 26, 6, 'Predicted 0/1', kind='lstm')
for a, b in [(b_e1, b_r1), (b_e2, b_r2), (b_r1, b_l1), (b_r2, b_l2), (b_l1, b_o1), (b_l2, b_o2)]:
    line(ax, [a['bot'], b['top']])
line(ax, [b_r['bot'], (50, y - 7.5), (25, y - 7.5), b_e1['top']]); line(ax, [(50, y - 7.5), (75, y - 7.5), b_e2['top']])
# confusion grid
gx, gy = 37, y - 70
ax.text(50, gy + 3.2, 'Compare the two columns', ha='center', fontsize=10.5, weight='bold', color=INK)
cells = [('TN', 'both 0', '#f1f0ec'), ('FP', 'only Predicted = 1', '#fde2e1'), ('FN', 'only Actual = 1', '#ececec'), ('TP', 'both 1', '#dff3e3')]
for i, (lab, sub, c) in enumerate(cells):
    cx, cy = gx + (i % 2) * 14, gy - 5 - (i // 2) * 8
    ax.add_patch(FancyBboxPatch((cx - 6.5, cy - 3.6), 13, 7.2, boxstyle='round,pad=0,rounding_size=0.8', fc=c, ec='#b5b2ab'))
    ax.text(cx, cy + 1.2, lab, ha='center', fontsize=11, weight='bold', color=INK); ax.text(cx, cy - 1.8, sub, ha='center', fontsize=7, color=INK2)
ax.text(gx - 9, gy - 5, 'Actual 0', ha='right', va='center', fontsize=8, color=INK2); ax.text(gx - 9, gy - 13, 'Actual 1', ha='right', va='center', fontsize=8, color=INK2)
ax.text(gx, gy - 18.5, 'Predicted 0', ha='center', fontsize=8, color=INK2); ax.text(gx + 14, gy - 18.5, 'Predicted 1', ha='center', fontsize=8, color=INK2)
ax.text(70, gy - 9, 'F1 = 2·P·R / (P + R)\nP = TP / (TP + FP)\nR = TP / (TP + FN)', fontsize=9, color=INK2, va='center', linespacing=1.6)
# why
ax.text(5, gy - 25, 'Why this works where the previous rules did not', fontsize=10.5, weight='bold', color=INK)
ax.text(5, gy - 29, '• An outage (actual ≈ 0 kW in full sun) is far below BOTH expected lines → Actual = 1 and Predicted = 1 → TP.\n'
        '• A cloud lowers the actual power AND both expected lines (they use the irradiation) → both 0 → TN.\n'
        '  Before, a cloudy morning was an "anomaly" because it was unusual for the time of day.\n'
        '• Before, the predicted line was judged alone; it only follows the weather, so it could never show an outage (FN).',
        fontsize=8.8, color=INK2, va='top', linespacing=1.6)
pages.append(('3_anomaly_definition', fig))

# ================================================================ page 4: experiment healthy vs all
fig, ax, top = page('4. Experiment — train the LSTM with faults or without?',
                    'Two LSTMs that differ ONLY in their training targets; everything else is identical.')
y = top - 17
b_s = box(ax, 50, y, 70, 7, 'Same data, windows (48), model, seed, epochs, early stopping', 'same anomaly rule, same ground truth (PVWatts), same k values, same readings')
b_a = box(ax, 27, y - 14, 40, 9, 'A. Healthy readings only', 'faulty readings (PVWatts check, 3σ)\nleft out of training + validation targets', kind='lstm', new='MAIN')
b_b = box(ax, 73, y - 14, 40, 9, 'B. All readings', 'faults included\n(as in the previous version)', kind='common')
b_ra = box(ax, 27, y - 28, 40, 7, 'Best k on training → F1', kind='anomaly')
b_rb = box(ax, 73, y - 28, 40, 7, 'Best k on training → F1', kind='anomaly')
line(ax, [b_s['bot'], (50, y - 6.5), (27, y - 6.5), b_a['top']]); line(ax, [(50, y - 6.5), (73, y - 6.5), b_b['top']])
line(ax, [b_a['bot'], b_ra['top']]); line(ax, [b_b['bot'], b_rb['top']])
# bars
bx0, by0, bh = 12, y - 72, 25
labels = {'Healthy readings only (main)': ('A. healthy only', '#4b3fa3'), 'All readings, faults included': ('B. all readings', '#b8b4ab')}
ax.plot([bx0 - 2, 94], [by0, by0], color='#b5b2ab', lw=1)
for j, s in enumerate(['Training', 'Validation', 'Testing']):
    for i, (lab, (short, c)) in enumerate(labels.items()):
        r = exp1[(exp1['LSTM trained on'] == lab) & (exp1['Split'] == s)].iloc[0]
        x = bx0 + j * 28 + i * 10
        ax.add_patch(FancyBboxPatch((x, by0), 8, bh * r.F1, boxstyle='square,pad=0', fc=c, ec='none'))
        ax.text(x + 4, by0 + bh * r.F1 + 1, f'{r.F1:.3f}', ha='center', fontsize=8.5, color=INK, weight='bold')
        ax.text(x + 4, by0 - 2.2, f'k={r.k:g}', ha='center', fontsize=7, color=MUTED)
    ax.text(bx0 + j * 28 + 9, by0 - 5.5, s, ha='center', fontsize=9.5, color=INK2)
ax.text(bx0 - 2, by0 + bh + 5, 'F1 with each model\'s best k (chosen on training)', fontsize=10.5, weight='bold', color=INK)
for i, (lab, (short, c)) in enumerate(labels.items()):
    ax.add_patch(FancyBboxPatch((62 + i * 17, by0 + bh + 4), 2.5, 2, boxstyle='square,pad=0', fc=c, ec='none'))
    ax.text(65.5 + i * 17, by0 + bh + 5, short, fontsize=8, va='center', color=INK2)
ax.text(5, by0 - 12, 'Why the healthy-trained model wins', fontsize=10.5, weight='bold', color=INK)
ax.text(5, by0 - 16, '• The LSTM only sees the weather, so it cannot know when an inverter is broken. Trained on faults,\n'
        '  it learns the AVERAGE of working and broken inverters: its line drops below healthy power\n'
        f'  (plant 4136001: {P2_FAULTY:.0%} of its training readings are faulty).\n'
        '• Then a real fault looks "less abnormal" (smaller gap) and normal readings look "too high" → more FN and FP.\n'
        '• This is the normal-behaviour-model principle: train on fault-free data so that faults stand out\n'
        '  (Malhotra et al. 2015; Hundman et al. 2018; Chesterman et al. 2023).',
        fontsize=8.8, color=INK2, va='top', linespacing=1.6)
pages.append(('4_experiment_healthy_vs_all', fig))

# ================================================================ page 5: before vs after
fig, ax, top = page('5. Previous version → v2: what changed, what stayed',
                    'Steps the colleague built are kept; only the anomaly rule and the training data of the LSTM change.')
rows = [
    ('Preprocessing, split 70/15/15', 'merge, fill ≤ 1 h, night removed, scaling', 'same', False),
    ('Window', '48 readings (12 h), chosen by validation loss', 'same (same experiment)', False),
    ('LSTM', '64 → 32 units, dropout 0.2, weather only', 'same', False),
    ('LSTM training data', 'all readings (faults included)', 'healthy readings only', True),
    ('Anomaly signal', 'each power line alone (change, or\nusual power at this time of day)', 'actual − expected power\n(each line alone)', True),
    ('Ground truth (Actual 0/1)', 'actual line vs its own history', 'actual vs physical expected\npower (PVWatts)', True),
    ('μ and σ', 'mean and std', 'median and 1.4826·MAD,\nper inverter, training days', True),
    ('Dawn / dusk (G < 0.2 kW/m²)', 'evaluated (many false alarms)', 'not evaluated; negative\npredictions set to 0', True),
    ('k values, best k on training', '1, 1.5, 2, 2.5, 3', 'same', False),
    ('Plots and six final files', 'Plots 1–7, files 1–6', 'same + expected line, Plot 8', False),
]
y = top - 20
ax.text(5, y, 'Step', fontsize=10, weight='bold', color=INK); ax.text(37, y, 'Previous version', fontsize=10, weight='bold', color=INK)
ax.text(70, y, 'v2', fontsize=10, weight='bold', color=NEW_C)
y -= 3
for step, old, new, changed in rows:
    h = 8.2 if ('\n' in old or '\n' in new) else 6
    ax.add_patch(FancyBboxPatch((4, y - h), 92, h - 0.6, boxstyle='round,pad=0,rounding_size=0.8',
                                fc='#fdf0e8' if changed else '#f4f3f0', ec='none'))
    ax.text(6, y - h / 2 - 0.3, step, fontsize=8.8, weight='bold', color=INK, va='center')
    ax.text(37, y - h / 2 - 0.3, old, fontsize=8.3, color=INK2, va='center', linespacing=1.3)
    ax.text(70, y - h / 2 - 0.3, new, fontsize=8.3, color=NEW_C if changed else INK2, va='center', linespacing=1.3,
            weight='bold' if changed else 'normal')
    y -= h
# result strip
y -= 5
ax.text(5, y, 'Effect on the numbers', fontsize=10.5, weight='bold', color=INK)
r1 = exp2[(exp2['Definition'].str.startswith('Rule 1')) & (exp2['Split'] == 'Testing')].iloc[0]
r2 = exp2[(exp2['Definition'].str.startswith('Rule 2')) & (exp2['Split'] == 'Testing')].iloc[0]
rn = exp2[(exp2['Definition'].str.startswith('NEW')) & (exp2['Split'] == 'Testing')].iloc[0]
txt = (f'Test outage readings (inverter ≤ 5 % of its neighbours): {int(rn["outage readings"])}\n'
       f'• Previous rule 1 (change):        in ground truth {int(r1["in Actual 0/1"])}, found by LSTM {int(r1["found by Predicted 0/1"])}   · test F1 {r1.F1:.3f}\n'
       f'• Previous rule 2 (time of day):  in ground truth {int(r2["in Actual 0/1"])}, found by LSTM {int(r2["found by Predicted 0/1"])}   · test F1 {r2.F1:.3f}\n'
       f'• v2 (actual − expected):          in ground truth {int(rn["in Actual 0/1"])}, found by LSTM {int(rn["found by Predicted 0/1"])}   · test F1 {rn.F1:.3f}')
ax.text(5, y - 3.5, txt, fontsize=8.8, color=INK2, va='top', linespacing=1.7, family='DejaVu Sans Mono')
ax.text(5, y - 20, 'F1 values of different rules are not strictly comparable (each rule has its own ground truth);\n'
        'the outage counts show what each ground truth contains and what the LSTM finds.', fontsize=8, color=MUTED, va='top', linespacing=1.5)
pages.append(('5_before_after', fig))

# ================================================================ page 6: notebook map
fig, ax, top = page('6. Notebook map — from → to', 'Run in this order; each notebook reads the outputs of the one before (Colab: upload the zip).')
nbs = [
    ('Notebook 1', 'Data preprocessing', 'unchanged', '4 Kaggle CSV files', 'step1_outputs: clean_data.csv,\nscalers.pkl, split + window tables', 'common'),
    ('Notebook 2', 'Window size & LSTM training', 'PVWatts expected power, healthy\nreadings, 2 LSTMs (healthy / all)', 'step1_outputs', 'step2_outputs: prediction_{split}.csv\n(actual, 2 LSTM lines, PVWatts, weather),\n2 models, loss curves, RMSE tables', 'lstm'),
    ('Notebook 3', 'Anomaly definition', 'actual − expected outside μ ± kσ\n(median / MAD, training days)', 'step2_outputs', 'step3_outputs: {split}_with_anomalies.csv\n(0/1 for every k and line),\nlimits_mu_sigma.csv', 'anomaly'),
    ('Notebook 4', 'Evaluation', 'F1 + confusion matrix per k, best k\non training; Experiments 1 and 2', 'step3_outputs', 'step4_outputs: A_plain, B_direction,\nexperiments (healthy vs all, previous rules,\noutage check, fault events)', 'anomaly'),
    ('Notebook 5', 'Plots & final files', 'same plots + expected line,\nPlot 8 (previous report days)', 'step3 + step4', 'final_outputs.zip: plots 1–8,\nsix final CSV files, summaries', 'result'),
]
y = top - 20
for i, (nb, name, change, inp, out, kind) in enumerate(nbs):
    cy = y - i * 22
    b = box(ax, 50, cy, 34, 12, nb, name, kind=kind, fs=11, sfs=9, new=None if change == 'unchanged' else 'CHANGED')
    ax.text(50, cy - 8.8, change, ha='center', va='top', fontsize=7.5, color=NEW_C if change != 'unchanged' else MUTED, linespacing=1.3)
    ax.text(30, cy, 'in: ' + inp, ha='right', va='center', fontsize=7.8, color=INK2)
    ax.text(70, cy, out, ha='left', va='center', fontsize=7.5, color=INK2, linespacing=1.35)
    if i < len(nbs) - 1:
        line(ax, [(50, cy - 6 - 6.8), (50, cy - 22 + 6)])
pages.append(('6_notebook_map', fig))

with PdfPages(PDF) as pdf:
    for name, f in pages:
        pdf.savefig(f)
        f.savefig(os.path.join(PNG, f'{name}.png'), dpi=160)
        plt.close(f)
print('wrote', PDF, 'and', len(pages), 'PNG pages')
