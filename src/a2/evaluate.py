"""The one evaluation protocol everyone reports with.

Task: 3-class fit prediction (small / fit / large). ~74% of rentals are "fit",
so plain accuracy rewards always predicting "fit". Macro-F1 (the headline
metric) weights each class equally; per-class F1 shows where a model fails.

Owner: B.
"""
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
    """{label: F1}. A class that is never predicted and never present scores 0."""
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


def report(name, y_true, y_pred, labels=FIT_LABELS):
    """Return one results-table row: model, Macro-F1, Accuracy, and F1 per class."""
    row = {"model": name, "Macro-F1": macro_f1(y_true, y_pred, labels), "Accuracy": accuracy(y_true, y_pred)}
    row.update({f"F1 {c}": s for c, s in f1_per_class(y_true, y_pred, labels).items()})
    return row


def results_table(rows):
    """Turn a list of report() rows into a DataFrame, best Macro-F1 first."""
    import pandas as pd

    return pd.DataFrame(rows).sort_values("Macro-F1", ascending=False).reset_index(drop=True)
