"""Baselines and models. Every model follows the same interface:

    model = SomeModel(...).fit(train_rows)
    preds = model.predict(rows)        # one prediction per row

Owners: B (baselines + course models), C (stronger model).
"""
from collections import defaultdict

from a2.data import ITEM_KEY, TARGET_KEY, USER_KEY


class GlobalMean:
    """Trivial baseline: always predict the training-set average."""

    def fit(self, rows):
        ys = [r[TARGET_KEY] for r in rows]
        if not ys:
            raise ValueError("cannot fit on an empty training set")
        self.mean_ = sum(ys) / len(ys)
        return self

    def predict(self, rows):
        return [self.mean_ for _ in rows]


class _GroupMean:
    key = None

    def __init__(self, shrinkage=0.0):
        # shrinkage pulls small groups towards the global mean: (sum + k*mu) / (n + k)
        self.shrinkage = shrinkage

    def fit(self, rows):
        self.global_ = GlobalMean().fit(rows)
        sums, counts = defaultdict(float), defaultdict(int)
        for r in rows:
            sums[r[self.key]] += r[TARGET_KEY]
            counts[r[self.key]] += 1
        mu, k = self.global_.mean_, self.shrinkage
        self.means_ = {g: (sums[g] + k * mu) / (counts[g] + k) for g in sums}
        return self

    def predict(self, rows):
        mu = self.global_.mean_
        return [self.means_.get(r[self.key], mu) for r in rows]


class ItemMean(_GroupMean):
    """Baseline: the item's average rating (global mean for unseen items)."""

    key = ITEM_KEY


class UserMean(_GroupMean):
    """Baseline: the user's average rating (global mean for unseen users)."""

    key = USER_KEY


# TODO(B): latent-factor / similarity-based model from the course.
# TODO(C): stronger model (e.g. text features, item2vec, temporal, BPR...).
