"""The one evaluation protocol everyone reports with.

Task: 3-class fit prediction (small / fit / large). ~74% of rentals are "fit",
so plain accuracy rewards always predicting "fit".

- Avg AUC (mean one-vs-rest AUC over the 3 classes) is threshold-free and is
  the metric used by Misra et al. (RecSys 2018), so we can compare to them.
- Macro-F1 weights each class equally and measures the hard decisions a
  retailer would act on. Decisions come from class-weighted argmax, with the
  weights tuned on the validation set only (tune_weights).
- Accuracy and per-class F1 are reported for context.

Probability arrays always have columns in FIT_LABELS order: small, fit, large.
"""
import itertools
import math

import numpy as np

from a2.data import FIT_LABELS


def _check(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} labels vs {len(y_pred)} predictions")
    if not y_true:
        raise ValueError("cannot evaluate on an empty set")
    return y_true, y_pred


def accuracy(y_true, y_pred):
    y_true, y_pred = _check(y_true, y_pred)
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def f1_per_class(y_true, y_pred, labels=FIT_LABELS):
    """{label: F1}. A class that is never predicted correctly scores 0."""
    y_true, y_pred = _check(y_true, y_pred)
    scores = {}
    for c in labels:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        scores[c] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return scores


def macro_f1(y_true, y_pred, labels=FIT_LABELS):
    scores = f1_per_class(y_true, y_pred, labels)
    return sum(scores.values()) / len(scores)


def average_auc(y_true, proba, labels=FIT_LABELS):
    """Mean one-vs-rest ROC AUC across classes (the paper's 'average AUC')."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(y_true))
    proba = np.asarray(proba, dtype=float)
    if proba.shape != (len(y), len(labels)):
        raise ValueError(f"proba must have shape ({len(y)}, {len(labels)}), got {proba.shape}")
    return float(np.mean([roc_auc_score(y == c, proba[:, k]) for k, c in enumerate(labels)]))


def decide(proba, weights=None, labels=FIT_LABELS):
    """Hard labels from probabilities: argmax of proba * weights."""
    proba = np.asarray(proba, dtype=float)
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    return [labels[k] for k in np.argmax(proba * w, axis=1)]


DEFAULT_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)


def tune_weights(y_true, proba, grid=DEFAULT_GRID, labels=FIT_LABELS):
    """Grid-search weights for the minority classes (weight of 'fit' fixed at 1)
    that maximise Macro-F1. Call this on the VALIDATION set only."""
    fit_idx = labels.index("fit")
    best, best_w = -1.0, None
    for ws in itertools.product(grid, repeat=len(labels) - 1):
        w = list(ws)
        w.insert(fit_idx, 1.0)
        score = macro_f1(y_true, decide(proba, w, labels), labels)
        if score > best:
            best, best_w = score, tuple(w)
    return best_w


def report(name, y_true, y_pred, proba=None, labels=FIT_LABELS):
    """One results-table row: model, Avg AUC (NaN without proba), Macro-F1, Accuracy, F1 per class."""
    row = {
        "model": name,
        "Avg AUC": average_auc(y_true, proba, labels) if proba is not None else math.nan,
        "Macro-F1": macro_f1(y_true, y_pred, labels),
        "Accuracy": accuracy(y_true, y_pred),
    }
    row.update({f"F1 {c}": s for c, s in f1_per_class(y_true, y_pred, labels).items()})
    return row


def results_table(rows):
    """Turn a list of report() rows into a DataFrame, best Macro-F1 first."""
    import pandas as pd

    return pd.DataFrame(rows).sort_values("Macro-F1", ascending=False).reset_index(drop=True)
