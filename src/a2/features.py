"""Feature construction for fit prediction.

Two groups of features:

1. Profile features - known before the rental from the customer's profile and the
   order: body measurements, rented size, category, body type, occasion.
2. History features - what the TRAINING data says about this item, this
   item-in-this-size ("product"), and this customer: how often they were
   small/large, the customer's usual size, and how this customer's body compares
   with people for whom this product fit.

History features for training rows are computed out-of-fold (K folds): a row's
features only use the other folds, so its own label never leaks into them and
training rows look like test rows (e.g. a customer with one rental has no history).

Deliberately NOT used: `rating`, `review_text`, `review_summary`, `review_date`.
They are written after the rental, so a real system could not use them to
predict fit beforehand (see models.TextFit for the post-hoc comparison).
"""
import numpy as np
import pandas as pd

CUP_ORDER = {"aa": 0, "a": 1, "b": 2, "c": 3, "d": 4, "d+": 5, "dd": 5, "ddd/e": 6,
             "f": 7, "g": 8, "h": 9, "i": 10, "j": 11}

NUMERIC = ["weight_lbs", "height_in", "bust_band", "cup_num", "age", "size", "bmi"]
MISSING_FLAGS = {"weight_missing": "weight_lbs", "height_missing": "height_in",
                 "bust_missing": "bust_band", "age_missing": "age"}
CATEGORICAL = ["category", "body type", "rented for"]
GROUPS = {"item": ["item_id"], "prod": ["item_id", "size"], "user": ["user_id"]}
BODY = {"weight_lbs": "weight_vs_prod_fit", "height_in": "height_vs_prod_fit", "bust_band": "bust_vs_prod_fit"}


def cup_to_number(cup):
    """Ordinal cup size (aa=0 ... j=11); unknown or missing -> None."""
    return CUP_ORDER.get(cup) if cup else None


def _frame(rows):
    df = pd.DataFrame({
        "user_id": [r.get("user_id") for r in rows],
        "item_id": [r.get("item_id") for r in rows],
        "fit": [r.get("fit") for r in rows],
        **{c: pd.to_numeric(pd.Series([r.get(c) for r in rows], dtype=object), errors="coerce")
           for c in ["weight_lbs", "height_in", "bust_band", "age", "size"]},
        "cup_num": pd.to_numeric(pd.Series([cup_to_number(r.get("bust_cup")) for r in rows], dtype=object)),
        **{c: [r.get(c) if r.get(c) is not None else "missing" for r in rows] for c in CATEGORICAL},
    })
    df["bmi"] = 703 * df["weight_lbs"] / df["height_in"] ** 2
    return df


def _history(hist, target, smoothing):
    """History features for `target` rows computed only from `hist` rows."""
    out = pd.DataFrame(index=target.index)
    prior = {c: float((hist["fit"] == c).mean()) for c in ("small", "large")}
    h = hist.assign(small=(hist["fit"] == "small").astype(float), large=(hist["fit"] == "large").astype(float))

    for name, keys in GROUPS.items():
        g = h.groupby(keys).agg(n=("fit", "size"), small=("small", "sum"), large=("large", "sum")).reset_index()
        m = target[keys].merge(g, on=keys, how="left").set_index(target.index)
        n = m["n"].fillna(0.0)
        out[f"{name}_n"] = n
        for c in ("small", "large"):
            p = (m[c] + smoothing * prior[c]) / (n + smoothing)
            out[f"{name}_p_{c}"] = p.where(n > 0)

    for name, keys in (("user", ["user_id"]), ("item", ["item_id"])):
        mean_size = h.groupby(keys)["size"].mean().rename("mean_size").reset_index()
        m = target[keys].merge(mean_size, on=keys, how="left").set_index(target.index)
        out[f"size_vs_{name}"] = target["size"] - m["mean_size"]

    fit_body = h[h["fit"] == "fit"].groupby(GROUPS["prod"])[list(BODY)].mean().reset_index()
    m = target[GROUPS["prod"]].merge(fit_body, on=GROUPS["prod"], how="left").set_index(target.index)
    for col, feat in BODY.items():
        out[feat] = target[col] - m[col]
    return out


class FeatureBuilder:
    def __init__(self, n_folds=5, smoothing=10.0, min_category_count=50, seed=0):
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.min_category_count = min_category_count
        self.seed = seed

    # ----- profile features -----
    def _profile(self, df):
        out = df[NUMERIC].astype(float).copy()
        for flag, col in MISSING_FLAGS.items():
            out[flag] = df[col].isna().astype(float)
        for col in CATEGORICAL:
            vals = df[col].where(df[col].isin(self.vocab_[col]), "other")
            for v in self.vocab_[col] + ["other"]:
                out[f"{col}={v}"] = (vals == v).astype(float)
        return out

    def fit_transform(self, rows):
        if not rows:
            raise ValueError("cannot build features from an empty training set")
        df = _frame(rows)
        self.train_ = df
        self.vocab_ = {}
        for col in CATEGORICAL:
            counts = df[col].value_counts()
            self.vocab_[col] = sorted(counts[counts >= self.min_category_count].index)
        profile = self._profile(df)
        self.profile_columns_ = list(profile.columns)

        folds = np.random.default_rng(self.seed).permutation(len(df)) % self.n_folds
        parts = []
        for k in range(self.n_folds):
            in_fold = folds == k
            if in_fold.any():
                parts.append(_history(df[~in_fold], df[in_fold], self.smoothing))
        history = pd.concat(parts).loc[df.index]
        X = pd.concat([profile, history], axis=1)
        self.columns_ = list(X.columns)
        return X

    def transform(self, rows):
        df = _frame(rows)
        X = pd.concat([self._profile(df), _history(self.train_, df, self.smoothing)], axis=1)
        return X[self.columns_]

