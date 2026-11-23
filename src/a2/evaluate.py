"""The one evaluation protocol everyone reports with.

Owner: B. If the task changes (e.g. to classification/ranking), change the
metrics here, not in individual notebooks.
"""
import math


def _check(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} labels vs {len(y_pred)} predictions")
    if not y_true:
        raise ValueError("cannot evaluate on an empty set")
    return y_true, y_pred


def mse(y_true, y_pred):
    y_true, y_pred = _check(y_true, y_pred)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    y_true, y_pred = _check(y_true, y_pred)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def accuracy(y_true, y_pred):
    y_true, y_pred = _check(y_true, y_pred)
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


METRICS = {"MSE": mse, "RMSE": rmse, "MAE": mae}


def report(name, y_true, y_pred, metrics=METRICS):
    """Return one results-table row: {"model": name, "MSE": ..., ...}."""
    row = {"model": name}
    for metric_name, fn in metrics.items():
        row[metric_name] = fn(y_true, y_pred)
    return row


def results_table(rows):
    """Turn a list of report() rows into a pandas DataFrame sorted by the first metric."""
    import pandas as pd

    df = pd.DataFrame(rows)
    first_metric = [c for c in df.columns if c != "model"][0]
    return df.sort_values(first_metric).reset_index(drop=True)
