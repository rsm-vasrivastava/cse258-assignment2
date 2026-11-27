"""Models for fit prediction. Every model follows the same interface:

    model = SomeModel(...).fit(train_rows)
    proba = model.predict_proba(rows)   # (n, 3), columns = FIT_LABELS = small, fit, large
    preds = model.predict(rows)         # hard labels (argmax, or argmax * self.weights)

Decision weights for Macro-F1 are tuned on the validation set with
a2.evaluate.tune_weights and stored on the model as `model.weights`.
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from a2.data import FIT_LABELS, ITEM_KEY, TARGET_KEY, USER_KEY
from a2.evaluate import decide
from a2.features import FeatureBuilder

LABEL_INDEX = {c: k for k, c in enumerate(FIT_LABELS)}


def _y_codes(rows):
    return np.array([LABEL_INDEX[r[TARGET_KEY]] for r in rows])


class _ProbModel:
    weights = None

    def predict(self, rows):
        return decide(self.predict_proba(rows), self.weights)


# =============================== baselines ===============================

class MajorityClass(_ProbModel):
    """Trivial baseline: always predict the most common training label ("fit").
    Its probabilities are the class priors, so its Avg AUC is exactly 0.5."""

    def fit(self, rows):
        counts = Counter(r[TARGET_KEY] for r in rows)
        if not counts:
            raise ValueError("cannot fit on an empty training set")
        self.label_ = counts.most_common(1)[0][0]
        total = sum(counts.values())
        self.prior_ = np.array([counts[c] / total for c in FIT_LABELS])
        return self

    def predict(self, rows):
        return [self.label_ for _ in rows]

    def predict_proba(self, rows):
        return np.tile(self.prior_, (len(rows), 1))


class _GroupMajority(_ProbModel):
    key = None

    def __init__(self, min_count=1, smoothing=5.0):
        # groups seen fewer than min_count times fall back to the global majority
        self.min_count = min_count
        self.smoothing = smoothing

    def fit(self, rows):
        self.fallback_ = MajorityClass().fit(rows)
        by_group = defaultdict(Counter)
        for r in rows:
            by_group[r[self.key]][r[TARGET_KEY]] += 1
        pref = {"fit": 0, "small": 1, "large": 2}  # tie-break: fit > small > large
        kept = {g: c for g, c in by_group.items() if sum(c.values()) >= self.min_count}
        self.labels_ = {g: min(c, key=lambda lab: (-c[lab], pref[lab])) for g, c in kept.items()}
        prior, k = self.fallback_.prior_, self.smoothing
        self.proba_ = {
            g: (np.array([c[lab] for lab in FIT_LABELS]) + k * prior) / (sum(c.values()) + k)
            for g, c in kept.items()
        }
        return self

    def predict(self, rows):
        if self.weights is not None:
            return super().predict(rows)
        default = self.fallback_.label_
        return [self.labels_.get(r[self.key], default) for r in rows]

    def predict_proba(self, rows):
        prior = self.fallback_.prior_
        return np.array([self.proba_.get(r[self.key], prior) for r in rows])


class ItemMajority(_GroupMajority):
    """Baseline: the item's fit history (does this dress run small/large?)."""

    key = ITEM_KEY


class UserMajority(_GroupMajority):
    """Baseline: the customer's fit history (does she usually size down/up?)."""

    key = USER_KEY


# ======================= 1-LV latent size model =======================
# Reproduces the "1-LV" baseline of Sembium et al. (RecSys 2017), as used by
# Misra et al. (RecSys 2018): each customer c has a true size u_c, each product
# p = (item, size) has a true size v_p, and z = u_c - v_p. Larger z means the
# customer is bigger than the product, i.e. the product runs "small".
# Ordinal logistic link with thresholds theta1 < theta2:
#   P(large) = s(theta1 - z),  P(fit) = s(theta2 - z) - s(theta1 - z),  P(small) = 1 - s(theta2 - z)

def _ordinal_proba(z, theta1, theta2):
    sa, sb = expit(theta1 - z), expit(theta2 - z)
    return np.column_stack([1 - sb, sb - sa, sa])  # small, fit, large


def _ordinal_loss_and_grad(params, ui, pi, y, n_u, n_p, reg):
    """Mean negative log-likelihood + reg * mean squared latents, and its gradient.
    params = [u (n_u), v (n_p), theta1, log(theta2 - theta1)]. y codes: 0 small, 1 fit, 2 large."""
    u, v = params[:n_u], params[n_u:n_u + n_p]
    theta1, delta = params[-2], params[-1]
    gap = np.exp(delta)
    z = u[ui] - v[pi]
    a, b = theta1 - z, theta1 + gap - z
    sa, sb = expit(a), expit(b)
    n = len(y)

    small, fit, large = y == 0, y == 1, y == 2
    nll = np.empty(n)
    nll[large] = np.logaddexp(0, -a[large])             # -log s(a)
    nll[small] = np.logaddexp(0, b[small])              # -log(1 - s(b))
    diff = np.maximum(sb[fit] - sa[fit], 1e-12)
    nll[fit] = -np.log(diff)

    d_a, d_b = np.zeros(n), np.zeros(n)
    d_a[large] = -(1 - sa[large])
    d_b[small] = sb[small]
    d_a[fit] = sa[fit] * (1 - sa[fit]) / diff
    d_b[fit] = -sb[fit] * (1 - sb[fit]) / diff
    d_z = -(d_a + d_b)

    loss = nll.sum() / n + reg * (u @ u + v @ v) / n
    grad = np.empty_like(params)
    grad[:n_u] = np.bincount(ui, d_z, n_u) / n + 2 * reg * u / n
    grad[n_u:n_u + n_p] = -np.bincount(pi, d_z, n_p) / n + 2 * reg * v / n
    grad[-2] = (d_a.sum() + d_b.sum()) / n
    grad[-1] = d_b.sum() * gap / n
    return loss, grad


class LatentSize(_ProbModel):
    """1-LV: one latent 'true size' per customer and per product (item, size)."""

    def __init__(self, reg=1.0, max_iter=500):
        self.reg = reg
        self.max_iter = max_iter

    @staticmethod
    def _product(r):
        return (r[ITEM_KEY], r.get("size"))

    def fit(self, rows):
        self.users_ = {u: k for k, u in enumerate(dict.fromkeys(r[USER_KEY] for r in rows))}
        self.products_ = {p: k for k, p in enumerate(dict.fromkeys(self._product(r) for r in rows))}
        ui = np.array([self.users_[r[USER_KEY]] for r in rows])
        pi = np.array([self.products_[self._product(r)] for r in rows])
        y = _y_codes(rows)
        n_u, n_p = len(self.users_), len(self.products_)
        # thresholds start at the empirical cumulative logits
        p_large, p_fit = (y == 2).mean(), (y == 1).mean()
        t1 = np.log(p_large / (1 - p_large))
        t2 = np.log((p_large + p_fit) / (1 - p_large - p_fit))
        x0 = np.concatenate([np.zeros(n_u + n_p), [t1, np.log(t2 - t1)]])
        res = minimize(_ordinal_loss_and_grad, x0, args=(ui, pi, y, n_u, n_p, self.reg),
                       jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.u_, self.v_ = res.x[:n_u], res.x[n_u:n_u + n_p]
        self.theta_ = (res.x[-2], res.x[-2] + np.exp(res.x[-1]))
        self.converged_ = res.success
        return self

    def score(self, rows):
        """z = u_c - v_p; unseen customers/products get 0 (the average)."""
        u = np.array([self.u_[self.users_[r[USER_KEY]]] if r[USER_KEY] in self.users_ else 0.0 for r in rows])
        v = np.array([self.v_[self.products_[p]] if (p := self._product(r)) in self.products_ else 0.0 for r in rows])
        return u - v

    def predict_proba(self, rows):
        return _ordinal_proba(self.score(rows), *self.theta_)


# ======================= feature-based models =======================

def _observed(X, cols):
    """Drop columns with no observed values in training (nothing to learn from)."""
    return [c for c in cols if X[c].notna().any()]


def _proba_in_label_order(clf, X):
    p = clf.predict_proba(X)
    order = [list(clf.classes_).index(k) for k in range(len(FIT_LABELS))]
    return p[:, order]


class LogisticFit(_ProbModel):
    """Multinomial logistic regression. profile_only=True uses only what's on the
    customer's profile + order; False also adds the history features."""

    def __init__(self, profile_only=True, C=1.0):
        self.profile_only = profile_only
        self.C = C

    def fit(self, rows):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.features_ = FeatureBuilder()
        X = self.features_.fit_transform(rows)
        cols = self.features_.profile_columns_ if self.profile_only else self.features_.columns_
        self.cols_ = _observed(X, cols)
        self.clf_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=self.C, max_iter=2000))
        self.clf_.fit(X[self.cols_].to_numpy(), _y_codes(rows))
        return self

    def predict_proba(self, rows):
        X = self.features_.transform(rows)[self.cols_].to_numpy()
        return _proba_in_label_order(self.clf_, X)


class GBMFit(_ProbModel):
    """Gradient-boosted trees on profile + history features (handles NaN natively)."""

    def __init__(self, learning_rate=0.05, max_iter=600, max_leaf_nodes=31, l2_regularization=1.0, seed=0):
        self.params = dict(learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                           l2_regularization=l2_regularization, random_state=seed,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)

    def fit(self, rows):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.features_ = FeatureBuilder()
        X = self.features_.fit_transform(rows)
        self.cols_ = _observed(X, self.features_.columns_)
        self.clf_ = HistGradientBoostingClassifier(**self.params).fit(X[self.cols_].to_numpy(), _y_codes(rows))
        return self

    def predict_proba(self, rows):
        return _proba_in_label_order(self.clf_, self.features_.transform(rows)[self.cols_].to_numpy())

    def feature_importance(self, rows, n_repeats=3, seed=0):
        """Permutation importance (drop in Avg AUC) on `rows`, e.g. the validation set."""
        import pandas as pd
        from a2.evaluate import average_auc

        X = self.features_.transform(rows)[self.cols_].to_numpy()
        y = [r[TARGET_KEY] for r in rows]
        base = average_auc(y, _proba_in_label_order(self.clf_, X))
        rng = np.random.default_rng(seed)
        drops = {}
        for j, col in enumerate(self.cols_):
            scores = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                scores.append(base - average_auc(y, _proba_in_label_order(self.clf_, Xp)))
            drops[col] = float(np.mean(scores))
        return pd.Series(drops).sort_values(ascending=False)


class TextFit(_ProbModel):
    """TF-IDF of the review (summary + text) -> logistic regression.

    POST-HOC ONLY: the review is written after the customer has worn the item,
    so this is not a usable pre-rental predictor. It shows how much fit
    information customers put into words (an upper-bound style comparison)."""

    def __init__(self, C=4.0, min_df=5, max_features=50000):
        self.C = C
        self.min_df = min_df
        self.max_features = max_features

    @staticmethod
    def _text(r):
        return f"{r.get('review_summary') or ''} . {r.get('review_text') or ''}"

    def fit(self, rows):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vec_ = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, max_features=self.max_features,
                                    sublinear_tf=True)
        X = self.vec_.fit_transform([self._text(r) for r in rows])
        self.clf_ = LogisticRegression(C=self.C, max_iter=2000).fit(X, _y_codes(rows))
        return self

    def predict_proba(self, rows):
        return _proba_in_label_order(self.clf_, self.vec_.transform([self._text(r) for r in rows]))

    def top_words(self, n=15):
        """Most indicative n-grams per class (largest coefficients)."""
        vocab = np.array(self.vec_.get_feature_names_out())
        return {FIT_LABELS[k]: list(vocab[np.argsort(-self.clf_.coef_[list(self.clf_.classes_).index(k)])[:n]])
                for k in range(len(FIT_LABELS))}
