"""Step 4 - compare Actual 0/1 with Predicted 0/1: confusion matrix, precision, recall, F1."""
import numpy as np
import pandas as pd

from . import config as C


def scores(actual, predicted):
    """actual / predicted: +1 rise, -1 drop, 0 normal. Anomaly = any non-zero value.

    'Wrong direction' counts readings that both lines flag but one says rise and the other drop;
    they are included in TP (both flagged an anomaly) and reported separately for information.
    """
    a = np.asarray(actual) != 0
    p = np.asarray(predicted) != 0
    tp, fp = int((a & p).sum()), int((~a & p).sum())
    fn, tn = int((a & ~p).sum()), int((~a & ~p).sum())
    wrong_dir = int((a & p & (np.asarray(actual) != np.asarray(predicted))).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'Wrong direction': wrong_dir,
            'Precision': round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}


def confusion(actual, predicted):
    a = (np.asarray(actual) != 0).astype(int)
    p = (np.asarray(predicted) != 0).astype(int)
    return np.array([[int(((a == i) & (p == j)).sum()) for j in (0, 1)] for i in (0, 1)])


def sweep(df, actual_prefix, pred_prefix, k_values=C.K_VALUES, by=None):
    """Scores for every split and every k (optionally per plant)."""
    rows = []
    groups = [('All', df)] if by is None else list(df.groupby(by))
    for name, g in groups:
        for s in C.SPLITS:
            gs = g[g['SPLIT'] == s]
            for k in k_values:
                r = {'Split': C.SPLIT_NAMES[s], 'k': k}
                if by is not None:
                    r[by] = name
                r.update(scores(gs[f'{actual_prefix}_k{k:g}'], gs[f'{pred_prefix}_k{k:g}']))
                rows.append(r)
    return pd.DataFrame(rows)


def best_k_from_training(table):
    """Highest training F1; on a tie the smaller k (fewer assumptions)."""
    tr = table[table['Split'] == 'Training'].sort_values(['F1', 'k'], ascending=[False, True])
    return float(tr.iloc[0]['k'])


def event_recall(df, actual_col, pred_col):
    """Event view (extra): an event = consecutive anomalous readings of one inverter in the ground truth.
    An event counts as found if the detector flags at least one of its readings."""
    found = total = 0
    for _, g in df.sort_values(['SOURCE_KEY', 'DATE_TIME']).groupby('SOURCE_KEY'):
        a = (g[actual_col].to_numpy() != 0)
        p = (g[pred_col].to_numpy() != 0)
        event_id = np.cumsum(np.r_[True, a[1:] != a[:-1]])
        for e in np.unique(event_id[a]):
            total += 1
            found += bool(p[event_id == e].any())
    return {'events': total, 'events found': found, 'event recall': round(found / total, 4) if total else 0.0}
