# Rebuild: python docs/v2/build/make_additions_document.py outputs docs/v2/SUNSHIELD_v2_additions_and_references.pdf  (needs reportlab, pillow; run the 5 notebooks first)
"""SUNSHIELD v2 — English document: what was added, why, and the research behind it.
usage: python doc_en.py <outputs_dir> <out_pdf>"""
import sys, os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT, PDF = sys.argv[1], sys.argv[2]
S2, S4, S5 = [os.path.join(OUT, f'step{i}_outputs') for i in (2, 4, 5)]
EXP = os.path.join(S4, 'experiments')

pdfmetrics.registerFont(TTFont('DV', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('DVB', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('DVI', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('DV', normal='DV', bold='DVB', italic='DVI', boldItalic='DVB')

INK, MUTED, ACCENT, LIGHT = colors.HexColor('#1f1e1c'), colors.HexColor('#5f5c56'), colors.HexColor('#b4471d'), colors.HexColor('#f3f1ed')
ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', fontName='DVB', fontSize=17, leading=21, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle('SUB', fontName='DV', fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10)
H2 = ParagraphStyle('H2', fontName='DVB', fontSize=12.5, leading=16, textColor=INK, spaceBefore=12, spaceAfter=5, keepWithNext=1)
H3 = ParagraphStyle('H3', fontName='DVB', fontSize=10.5, leading=14, textColor=ACCENT, spaceBefore=8, spaceAfter=3, keepWithNext=1)
P = ParagraphStyle('P', fontName='DV', fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=5, alignment=TA_LEFT)
B = ParagraphStyle('B', parent=P, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
CAP = ParagraphStyle('CAP', fontName='DVI', fontSize=8, leading=11, textColor=MUTED, spaceBefore=2, spaceAfter=8)
CELL = ParagraphStyle('CELL', fontName='DV', fontSize=8, leading=10.2, textColor=INK)
CELLB = ParagraphStyle('CELLB', parent=CELL, fontName='DVB')
BOX = ParagraphStyle('BOX', parent=P, backColor=colors.HexColor('#fbefe8'), borderPadding=(6, 7, 6, 7), spaceBefore=4, spaceAfter=9)

def para(t, st=P): return Paragraph(t, st)
def bullets(items): return [Paragraph(t, B, bulletText='•') for t in items]
def table(rows, widths, header=True, highlight_last=False, bold_col=None):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) or (bold_col is not None and j == bold_col and i > 0) else CELL)
             for j, c in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LINEBELOW', (0, 0), (-1, 0), 0.8, INK),
          ('LINEBELOW', (0, -1), (-1, -1), 0.5, MUTED), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
          ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT])]
    if highlight_last:
        st.append(('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#fff4d6')))
    t.setStyle(TableStyle(st))
    return t
def img(path, width_cm, caption=None):
    from PIL import Image as PI
    w, h = PI.open(path).size
    out = [Image(path, width=width_cm * cm, height=width_cm * cm * h / w)]
    if caption: out.append(para(caption, CAP))
    return [KeepTogether(out)]
f3 = lambda x: f'{x:.3f}'

# ---------------------------------------------------------------- numbers
resA = pd.read_csv(os.path.join(S4, 'A_plain', 'results_best_k.csv')).set_index('Split')
allk = pd.read_csv(os.path.join(S4, 'A_plain', 'metrics_all_k.csv'))
plant = pd.read_csv(os.path.join(S4, 'A_plain', 'results_best_k_per_plant.csv'))
exp1 = pd.read_csv(os.path.join(EXP, 'exp1_training_data_best_k.csv'))
exp1p = pd.read_csv(os.path.join(EXP, 'exp1_training_data_per_plant.csv'))
exp2 = pd.read_csv(os.path.join(EXP, 'exp2_previous_rules.csv'))
outc = pd.read_csv(os.path.join(EXP, 'outage_check.csv'))
events = pd.read_csv(os.path.join(EXP, 'fault_events.csv'))
sens = pd.read_csv(os.path.join(EXP, 'sensitivity.csv'))
met = pd.read_csv(os.path.join(S2, 'metrics.csv'))
share = pd.read_csv(os.path.join(S2, 'healthy_share.csv'))
K = float(resA['k'].iloc[0])
peer = pd.read_csv(os.path.join(EXP, 'peer_reference_problem.csv'), index_col=0)['value']
lim = pd.read_csv(os.path.join(OUT, 'step3_outputs', 'limits_mu_sigma.csv'))
ks_lo, ks_hi = K * lim['pred_sigma'].quantile(0.25), K * lim['actual_sigma'].quantile(0.75)
H, A = 'Healthy readings only (main)', 'All readings, faults included'
e1 = {(r['LSTM trained on'], r['Split']): r for _, r in exp1.iterrows()}
r1t = exp2[exp2['Definition'].str.startswith('Rule 1') & (exp2['Split'] == 'Testing')].iloc[0]
r2t = exp2[exp2['Definition'].str.startswith('Rule 2') & (exp2['Split'] == 'Testing')].iloc[0]
r2s = exp2[exp2['Definition'].str.startswith('Rule 2')].set_index('Split')
r1s = exp2[exp2['Definition'].str.startswith('Rule 1')].set_index('Split')
rnt = exp2[exp2['Definition'].str.startswith('NEW') & (exp2['Split'] == 'Testing')].iloc[0]
oc = outc.set_index(['Split', 'Line'])
def rmse(split, plant_id, model, subset):
    m = met[(met['split'] == split) & (met['plant'].astype(str) == str(plant_id)) & (met['trained on'] == model) & (met['scored on'] == subset)]
    return m['RMSE (kW)'].iloc[0]
sh = share.set_index(['PLANT_ID', 'SPLIT'])
removed_p2 = 1 - sh.loc[(4136001, 'train'), 'healthy share']
removed_p1 = 1 - sh.loc[(4135001, 'train'), 'healthy share']

story = []
story += [para('SUNSHIELD v2 — what we added to the LSTM anomaly detection, and why', H1),
          para('For the team. Built on the colleague\'s TARGET_SHIFT = 0 notebooks (1–5); preprocessing, window, LSTM, '
               'k values and plots are kept. Every number below comes from the notebooks in <font face="DVB">notebooks/v2/</font>; '
               'every change follows a published method (references at the end).', SUB)]

# ---------------------------------------------------------------- 1 summary
story.append(para('1. Summary', H2))
story.append(para(
    'The previous version reached a test F1 of 0.59–0.73 and missed the long outages of plant 4136001. Two things caused it: '
    '(1) the anomaly rule looked at each power line <i>alone</i>. The LSTM line only follows the weather, so it can never show an '
    'inverter that stops while the sun shines, and the actual line was flagged for clouds as well as for faults. '
    '(2) The LSTM was trained on readings that include faults, so it learned to predict too little power. '
    'v2 keeps Dr. Mai\'s workflow box for box and changes only what goes inside two boxes: the anomaly is now '
    '<b>the gap between the actual power and the expected power</b>, and the LSTM is <b>trained on healthy readings only</b>.'))
rows = [['', 'Previous (report, k = 1)', 'Previous (Notebook 3, k = 1)', f'v2 (k = {K:g})'],
        ['Training F1', f3(r2s.loc['Training', 'F1']), f3(r1s.loc['Training', 'F1']), f3(resA.loc['Training', 'F1'])],
        ['Validation F1', f3(r2s.loc['Validation', 'F1']), f3(r1s.loc['Validation', 'F1']), f3(resA.loc['Validation', 'F1'])],
        ['Testing F1', f3(r2s.loc['Testing', 'F1']), f3(r1s.loc['Testing', 'F1']), f'<b>{f3(resA.loc["Testing", "F1"])}</b>'],
        ['Test outage readings in ground truth', f'{int(r2t["in Actual 0/1"])} of {int(r2t["outage readings"])}',
         f'{int(r1t["in Actual 0/1"])} of {int(r1t["outage readings"])}', f'{int(rnt["in Actual 0/1"])} of {int(rnt["outage readings"])}'],
        ['… of those found by the LSTM line', f'{int(r2t["found by Predicted 0/1"])}', f'{int(r1t["found by Predicted 0/1"])}',
         f'<b>{int(rnt["found by Predicted 0/1"])}</b>']]
story.append(table(rows, [5.2 * cm, 3.9 * cm, 3.9 * cm, 3.4 * cm]))
story.append(para('"Previous (report)" = usual power at this time of day; "Previous (Notebook 3)" = change from the previous reading. '
                  'Outage reading = the inverter gives ≤ 5 % of its neighbours\' median power while the neighbours give > 100 kW '
                  '(an independent check that does not use any of the rules). Different rules have different ground truths, so '
                  'their F1 values are not strictly comparable; the outage rows show what each ground truth contains.', CAP))

# ---------------------------------------------------------------- 2 kept
story.append(para('2. What did not change', H2))
story += bullets([
    'Notebook 1 is the same code: merge, weather gaps ≤ 1 h filled, night (irradiation = 0) removed, 70/15/15 split by whole days, '
    'StandardScaler fitted on training only, windows inside unbroken runs (RUN_ID), TARGET_SHIFT = 0. Its output is identical to the previous run.',
    'The window-size experiment (8–48) is re-run unchanged and again selects <b>W = 48</b> (val MSE 0.1913).',
    'The LSTM (64 → 32 units, dropout 0.2, Adam 1e-3, batch 256, early stopping patience 10, seed 42) and its 3 weather inputs.',
    'Dr. Mai\'s workflow: k = 1, 1.5, 2, 2.5, 3; one Actual 0/1 and one Predicted 0/1 column per k; F1 + confusion matrix per k on '
    'training; best k from training only; the test uses that k. Version A (plain 0/1) is the main result, Version B (direction) is kept.',
    'Notebook 5: the same plots (1–7) and the six final CSV files with the same names and the columns used by app.py.'])

# ---------------------------------------------------------------- 3 diagnosis
story.append(para('3. Why the F1 was low and the outages were missed', H2))
story.append(para('3.1  The predicted line cannot show a fault', H3))
story.append(para(
    f'The LSTM sees only irradiation and temperatures. When an inverter stops at noon the weather is normal, so the LSTM line stays normal. '
    f'With a rule applied to the predicted line alone, the outage can only be an FN. Measured on the test days with the time-of-day rule: '
    f'the ground truth contains {int(r2t["in Actual 0/1"])} of the {int(r2t["outage readings"])} outage readings, the LSTM line flags '
    f'{int(r2t["found by Predicted 0/1"])}, and those only because the weather at that moment was unusual. This is the gap seen in the report\'s '
    f'appendix (8–9 June, plant 4136001). All LSTM anomaly-detection methods compare the actual value with the predicted value for '
    f'this reason (Malhotra et al. 2015; Hundman et al. 2018); in PV, De Benedetti et al. (2018) compare measured and predicted AC power.'))
story.append(para('3.2  The ground truth labelled weather, not faults', H3))
story.append(para(
    f'"Change from the previous reading" marks only the moment power drops: {int(r1t["in Actual 0/1"])} of {int(r1t["outage readings"])} '
    f'test outage readings are in its ground truth; the rest of each outage is "normal". "Usual power at this time of day" with k = 1 '
    f'flags about one reading in three (68-95-99.7 rule), including cloudy mornings when the inverter works normally (report, Finding 3).'))
story.append(para('3.3  The LSTM learned the faults', H3))
story.append(para(
    f'Plant 4136001 has many outages: {removed_p2:.1%} of its training readings fail the physical check below '
    f'(plant 4135001: {removed_p1:.1%}). A weather-only model trained on them learns the average of working and broken inverters, so its '
    f'line is too low (Plot 8: about 600 kW instead of about 1,100 kW at noon). On healthy plant-2 test readings its RMSE is '
    f'{rmse("test", 4136001, "all readings", "healthy readings"):.0f} kW; the same LSTM trained on healthy readings gets '
    f'{rmse("test", 4136001, "healthy only (main)", "healthy readings"):.0f} kW. The previous report\'s "irradiation explains only 31 % of '
    f'plant 4136001" comes from these outages: healthy inverters of both plants produce about the same power per unit of irradiation.'))

# ---------------------------------------------------------------- 4 additions
story.append(para('4. The additions — what, why, and the research behind each', H2))
story.append(para(
    '<b>The new anomaly definition, in one line:</b> for each line, r = actual − expected power; the reading is 1 if '
    'r &lt; μ − kσ or r &gt; μ + kσ, with μ = median and σ = 1.4826·MAD of r for this inverter on the training days. '
    'Actual 0/1 uses the physical expected power, Predicted 0/1 uses the LSTM line. Same rule, applied to each line separately.', BOX))

adds = [
    ('A1. The anomaly is the gap between actual and expected power (residual)',
     'Notebook 3. Both 0/1 columns use r = actual − expected instead of the shape of one line.',
     'An outage is far below both expected lines, so it is 1 in both columns (TP). A cloud lowers the actual power <i>and</i> the '
     'expected power, so it is 0 in both (TN). This removes the two problems of section 3.',
     'Prediction error of an LSTM trained on normal data is the anomaly signal: Malhotra et al. (2015), Hundman et al. (2018). '
     'PV: comparison of measured and predicted AC power and analysis of the residuals (De Benedetti et al. 2018); residuals between '
     'measured and model-predicted PV output monitored with a control chart with ±L·σ limits (Garoudja et al. 2017; Harrou et al. 2018).'),
    ('A2. Ground truth (Actual 0/1) from physics: PVWatts expected power',
     'Notebook 2, Step 3. P_exp = P_ref · G · (1 + γ (T_module − 25 °C)) with the measured irradiation G and module temperature, '
     'γ = −0.47 %/°C. P_ref of each inverter = median of AC / (G · temperature factor) on the training days with G ≥ 0.2 kW/m².',
     'The actual 0/1 needs a reference that is not the LSTM. A physical formula is independent of the model, needs no labels and is '
     'the standard expected-power model of NREL. The median makes P_ref robust to the outages in the training days.',
     'PVWatts DC model and its temperature coefficient: Dobos (2014). NREL\'s RdTools normalises measured PV output with the same '
     'PVWatts expected power. Physical model + residual + k·σ limits for PV faults: Garoudja et al. (2017), Harrou et al. (2018).'),
    ('A3. Robust μ and σ: median and 1.4826 · MAD, per inverter, training days only',
     'Notebooks 2 and 3. μ and σ of each line\'s residual are computed once per inverter from the training days; the same limits are '
     'used for validation and testing (as in Dr. Mai\'s workflow).',
     'Mean and standard deviation are pulled by the faults themselves (plant 4136001), which widens the limits and hides faults. '
     'Median and MAD are not. 1.4826·MAD equals σ for normally distributed data, so "μ ± kσ" keeps its meaning.',
     'Leys et al. (2013): use the median absolute deviation instead of the standard deviation. For PV arrays, Zhao et al. (2013) '
     'recommend the Hampel identifier (median/MAD) and the boxplot rule over the 3-sigma rule.'),
    ('A4. Readings with irradiation below 0.2 kW/m² are not evaluated; negative predictions are set to 0',
     'Notebooks 2–4. Dawn and dusk readings get 0 and are left out of F1 and the confusion matrices (column EVALUATED).',
     'There the expected power is a few kW and every model\'s relative error is large: the previous report (Finding 4) found the '
     'predicted line flagged in 36 % of the 05:00–05:45 readings. Negative power is impossible (option listed in the report).',
     'RdTools (NREL) filters plane-of-array irradiance outside 200–1200 W/m² (poa_filter defaults) before analysing performance.'),
    ('A5. The LSTM is trained on healthy readings only (normal-behaviour model)',
     f'Notebook 2, Step 5. A training or validation reading is healthy if its residual from PVWatts is inside median ± 3σ '
     f'(the classic 3-sigma limit, fixed before choosing k). The windows are unchanged; only unhealthy <i>targets</i> are left out '
     f'({removed_p1:.1%} of plant 4135001 and {removed_p2:.1%} of plant 4136001 training readings).',
     'The model should learn what a working inverter produces under this weather; then a fault stands out as a large gap. '
     'Experiment 1 below measures the effect.',
     'Malhotra et al. (2015) train the LSTM on non-anomalous data. Normal-behaviour models are trained on healthy data, not polluted by '
     'anomalies related to failures (Chesterman et al. 2023, review of NBM for wind turbines).'),
]
for title, what, why, research in adds:
    story.append(KeepTogether([para(title, H3),
        table([['What', what], ['Why', why], ['Research', research]], [2.1 * cm, 14.3 * cm], header=False)]))
    story.append(Spacer(1, 4))

story.append(KeepTogether([para('Alternative tested and not used: neighbours as the ground truth', H3), para(
    'Comparing an inverter with the median of the other inverters of its plant (the idea of SolarClique, Iyengar et al. 2018) was '
    f'tested first. In plant 4136001 many inverters are often down at the same moment, which pulls the median down: '
    f'{int(peer.iloc[0])} plant-2 readings are more than 20 % above their neighbours\' median; at those moments a median of '
    f'{peer.iloc[1]:.0f} inverters of the plant are down (otherwise {peer.iloc[2]:.0f}), and the same readings are normal compared with '
    f'the LSTM (median actual / LSTM = {peer.iloc[3]:.2f}). They would become false ground-truth anomalies. The physical reference does '
    f'not depend on the other inverters, so it was preferred. The neighbours are still used for the independent outage check '
    f'(Notebook 4, "Why the neighbours are not used").')]))

# ---------------------------------------------------------------- 5 experiment 1
story.append(para('5. Experiment 1 — train the LSTM with faults or without?', H2))
story.append(para('Two LSTMs that differ only in their training targets: same windows (48), architecture, seed, epochs, early stopping, '
                  'anomaly rule, ground truth, k values and readings. Each model gets its own best k from training, as in the workflow.'))
rows = [['LSTM trained on', 'best k', 'Training F1', 'Validation F1', 'Testing F1', 'Test precision', 'Test recall']]
for lab, nm in [(H, 'Healthy readings only (v2)'), (A, 'All readings, faults included (previous)')]:
    rows.append([nm, f'{e1[(lab, "Testing")]["k"]:g}'] + [f3(e1[(lab, s)]['F1']) for s in ['Training', 'Validation', 'Testing']]
                + [f3(e1[(lab, 'Testing')]['Precision']), f3(e1[(lab, 'Testing')]['Recall'])])
story.append(table(rows, [5.4 * cm, 1.4 * cm, 1.9 * cm, 2.1 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm]))
story.append(Spacer(1, 5))
pp = exp1p.set_index(['LSTM trained on', 'PLANT_ID'])
rows = [['Per plant (F1)', 'Training', 'Validation', 'Testing']]
for lab, nm in [(H, 'healthy only'), (A, 'all readings')]:
    for pid in [4135001, 4136001]:
        r = pp.loc[(lab, pid)]
        rows.append([f'Plant {pid}, {nm}', f3(r['Training']), f3(r['Validation']), f3(r['Testing'])])
story.append(table(rows, [6 * cm, 3 * cm, 3 * cm, 3 * cm]))
story.append(Spacer(1, 5))
rows = [['Independent check (no PVWatts, no k)', 'Training', 'Validation', 'Testing']]
for line, nm in [('LSTM healthy only (main)', 'Outages flagged — healthy-trained LSTM'), ('LSTM all readings', 'Outages flagged — LSTM trained on all readings')]:
    rows.append([nm] + [f'{int(oc.loc[(s, line), "flagged"])} / {int(oc.loc[(s, line), "peer-confirmed outage readings"])}' for s in ['Training', 'Validation', 'Testing']])
ev = events.set_index(['Split', 'LSTM'])
for line, nm in [('LSTM healthy only (main)', 'Fault events found — healthy-trained'), ('LSTM all readings', 'Fault events found — all readings')]:
    rows.append([nm] + [f'{int(ev.loc[(s, line), "found"])} / {int(ev.loc[(s, line), "events"])}' for s in ['Training', 'Validation', 'Testing']])
story.append(table(rows, [7.2 * cm, 3 * cm, 3 * cm, 3 * cm]))
story.append(para('Fault event = consecutive readings of one inverter on one day with Actual = 1 (same events for both models, k = '
                  f'{K:g}); found = at least one Predicted = 1 inside it.', CAP))
story.append(para(
    f'<b>Conclusion:</b> training on healthy readings raises the test F1 from {f3(e1[(A, "Testing")]["F1"])} to '
    f'{f3(e1[(H, "Testing")]["F1"])} and the validation F1 from {f3(e1[(A, "Validation")]["F1"])} to {f3(e1[(H, "Validation")]["F1"])}. '
    f'The gain comes from plant 4136001, where the faults had pulled the old LSTM line down; plant 4135001 has few faults and changes little. '
    'We therefore train on healthy readings.', BOX))
story += img(os.path.join(EXP, 'exp1_training_data.png'), 16.5, 'Left: F1 for every k for both models. Right: F1 with each model\'s best k (chosen on training).')

# ---------------------------------------------------------------- 6 results
story.append(para(f'6. Final results (Version A, plain 0/1, best k = {K:g} from training)', H2))
rows = [['Split', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1']]
for s in ['Training', 'Validation', 'Testing']:
    r = resA.loc[s]
    rows.append([s, f'{int(r.TP):,}', f'{int(r.FP):,}', f'{int(r.FN):,}', f'{int(r.TN):,}', f3(r.Precision), f3(r.Recall), f'<b>{f3(r.F1)}</b>'])
story.append(table(rows, [2.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.9 * cm, 2.1 * cm, 2.1 * cm, 1.9 * cm], highlight_last=True))
story.append(Spacer(1, 4))
rows = [['k', 'Training F1', 'Validation F1', 'Testing F1']]
pv = allk.pivot(index='k', columns='Split', values='F1')
for k, r in pv.iterrows():
    rows.append([f'{k:g}' + (' (best)' if k == K else ''), f3(r['Training']), f3(r['Validation']), f3(r['Testing'])])
story.append(table(rows, [3 * cm, 3 * cm, 3 * cm, 3 * cm]))
story.append(para('F1 rises with k up to 3, the end of the workflow\'s range; k = 3 is also the classic 3σ limit.', CAP))
rows = [['Per plant (best k)', 'Training F1', 'Validation F1', 'Testing F1', 'Test anomalies (TP + FN)']]
for pid in [4135001, 4136001]:
    g = plant[plant['PLANT_ID'] == pid].set_index('Split')
    rows.append([f'Plant {pid}', f3(g.loc['Training', 'F1']), f3(g.loc['Validation', 'F1']), f3(g.loc['Testing', 'F1']),
                 f'{int(g.loc["Testing", "TP"] + g.loc["Testing", "FN"])}'])
story.append(table(rows, [3.6 * cm, 2.6 * cm, 2.8 * cm, 2.6 * cm, 4.4 * cm]))
story.append(para('Plant 4135001 has few anomalies, so its precision rests on small numbers: most of its FP are readings where the '
                  'LSTM and the physical formula disagree by a little more than 3σ.', CAP))
story += img(os.path.join(S5, 'plots', '5_confusion_matrices_best_k.png'), 16.5, 'Confusion matrices with the best k (evaluated readings, irradiation ≥ 0.2 kW/m²).')
story += img(os.path.join(S5, 'plots', '8_previous_report_examples_new_method.png'), 16.5,
             'Plot 8 — the example days of the previous report with v2. Top: the outages of 8–9 June are now found (red dots inside the '
             'outage, triangles on the LSTM line at the same times); the dash-dot line is the previous LSTM trained on all readings. '
             'Bottom: the cloudy morning of 11 June is no longer an anomaly, because the expected power is low too.')

# ---------------------------------------------------------------- 7 sensitivity + limitations
story.append(para('7. Sensitivity check', H2))
rows = [['Variant (main LSTM kept)', 'best k', 'Training F1', 'Validation F1', 'Testing F1']]
for _, r in sens.iterrows():
    rows.append([r['Variant'], f'{r["best k"]:g}', f3(r['Training']), f3(r['Validation']), f3(r['Testing'])])
story.append(table(rows, [7.4 * cm, 1.6 * cm, 2.4 * cm, 2.6 * cm, 2.4 * cm]))
story.append(para('The temperature coefficient and the irradiance limit change F1 only in the second decimal. (The "main" row differs '
                  'from section 6 in the 3rd decimal because P_ref is re-fitted on the windowed readings in this check.)', CAP))
rvc = pd.read_csv(os.path.join(EXP, 'robust_vs_classic_ground_truth.csv'), index_col=0)
story.append(para(
    'The classic mean and standard deviation give a slightly higher F1, but for the wrong reason. The outages of plant 4136001 '
    'inflate its standard deviation, so the limits there become several times wider than in plant 4135001, and a partial loss of '
    '20–50 % counts as normal. Its ground truth is therefore smaller and easier to agree on:'))
rows = [[''] + list(rvc.columns)] + [[i] + [str(v) for v in r] for i, r in rvc.iterrows()]
story.append(table(rows, [8.4 * cm, 4 * cm, 4 * cm]))
story.append(para('Test days. With mean/std the ground truth loses part of the outages and most partial losses, so it is a weaker '
                  'detector even though the agreement (F1) is higher. We keep median/MAD (Leys et al. 2013; Zhao et al. 2013).', CAP))

story.append(para('8. Limitations', H2))
story += bullets([
    'There is no operator fault log. The ground truth is measured (physics), not hand-labelled; the neighbour-based outage check is '
    'an independent confirmation, not a label.',
    'The physical formula and the LSTM use the same weather sensor; if the sensor is wrong at a moment, both can be wrong together.',
    'Readings below 0.2 kW/m² (early morning, late evening) are not judged, so a fault that only happens then is not detected.',
    f'At k = {K:g} a gap must be larger than kσ (about {ks_lo:.0f}–{ks_hi:.0f} kW for most inverters) to count; small slow losses are out of scope.',
    '34 days of one season; P_ref, μ, σ and the LSTM should be re-fitted when more data is available.'])

# ---------------------------------------------------------------- refs
story.append(para('References', H2))
refs = [
    'Malhotra, P., Vig, L., Shroff, G., Agarwal, P. (2015). Long Short Term Memory Networks for Anomaly Detection in Time Series. ESANN 2015.',
    'Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T. (2018). Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. KDD 2018. arXiv:1802.04431.',
    'De Benedetti, M., Leonardi, F., Messina, F., Santoro, C., Vasilakos, A. (2018). Anomaly detection and predictive maintenance for photovoltaic systems. Neurocomputing 310, 59–68. doi:10.1016/j.neucom.2018.05.017.',
    'Garoudja, E., Harrou, F., Sun, Y., Kara, K., Chouder, A., Silvestre, S. (2017). Statistical fault detection in photovoltaic systems. Solar Energy 150, 485–499. doi:10.1016/j.solener.2017.04.043.',
    'Harrou, F., Sun, Y., Taghezouit, B., Saidi, A., Hamlati, M.-E. (2018). Reliable fault detection and diagnosis of photovoltaic systems based on statistical monitoring approaches. Renewable Energy 116, 22–37. doi:10.1016/j.renene.2017.09.048.',
    'Dobos, A. P. (2014). PVWatts Version 5 Manual. NREL/TP-6A20-62641. National Renewable Energy Laboratory.',
    'RdTools — open-source PV performance analysis library, National Renewable Energy Laboratory (rdtools.readthedocs.io): filtering.poa_filter (200–1200 W/m²), normalization.normalize_with_pvwatts.',
    'Leys, C., Ley, C., Klein, O., Bernard, P., Licata, L. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. Journal of Experimental Social Psychology 49(4), 764–766.',
    'Zhao, Y., Lehman, B., Ball, R., Mosesian, J., de Palma, J.-F. (2013). Outlier detection rules for fault detection in solar photovoltaic arrays. IEEE APEC 2013. doi:10.1109/APEC.2013.6520712.',
    'Chesterman, X., Verstraeten, T., Daems, P.-J., Nowé, A., Helsen, J. (2023). Overview of normal behavior modeling approaches for SCADA-based wind turbine condition monitoring demonstrated on data from operational wind farms. Wind Energy Science 8, 893–924. doi:10.5194/wes-8-893-2023.',
    'Iyengar, S., Lee, S., Sheldon, D., Shenoy, P. (2018). SolarClique: Detecting Anomalies in Residential Solar Arrays. ACM COMPASS 2018.',
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(r, ParagraphStyle('R', parent=P, fontSize=8.2, leading=11, leftIndent=14, firstLineIndent=-14), bulletText=None))

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('DV', 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, 'SUNSHIELD v2 — additions and references'); canvas.drawRightString(19 * cm, 1.2 * cm, f'{doc.page}')
    canvas.restoreState()

doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title='SUNSHIELD v2 — additions and references', author='SUNSHIELD team')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('wrote', PDF)
