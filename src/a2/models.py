"""Baselines and models for fit prediction. Every model follows the same interface:

    model = SomeModel(...).fit(train_rows)
    preds = model.predict(rows)        # one label from FIT_LABELS per row

Owners: B (baselines + course models), C (stronger model).
"""
from collections import Counter, defaultdict

from a2.data import ITEM_KEY, TARGET_KEY, USER_KEY


class MajorityClass:
    """Trivial baseline: always predict the most common training label ("fit")."""

    def fit(self, rows):
        counts = Counter(r[TARGET_KEY] for r in rows)
        if not counts:
            raise ValueError("cannot fit on an empty training set")
        self.label_ = counts.most_common(1)[0][0]
        return self

    def predict(self, rows):
        return [self.label_ for _ in rows]


class _GroupMajority:
    key = None

    def __init__(self, min_count=1):
        # groups seen fewer than min_count times fall back to the global majority
        self.min_count = min_count

    def fit(self, rows):
        self.fallback_ = MajorityClass().fit(rows)
        by_group = defaultdict(Counter)
        for r in rows:
            by_group[r[self.key]][r[TARGET_KEY]] += 1
        # ties broken by preference: fit > small > large
        pref = {"fit": 0, "small": 1, "large": 2}
        self.labels_ = {
            g: min(c, key=lambda lab: (-c[lab], pref.get(lab, 3)))
            for g, c in by_group.items()
            if sum(c.values()) >= self.min_count
        }
        return self

    def predict(self, rows):
        default = self.fallback_.label_
        return [self.labels_.get(r[self.key], default) for r in rows]


class ItemMajority(_GroupMajority):
    """Baseline: the item's most common fit label (does this dress run small/large?)."""

    key = ITEM_KEY


class UserMajority(_GroupMajority):
    """Baseline: the user's most common fit label (does this customer size down/up?)."""

    key = USER_KEY


# TODO(B): logistic regression on features (size, height, weight, bust, category, body type...).
# TODO(B): latent "true size" model in the spirit of Misra et al. 2018 (user/item size parameters).
# TODO(C): stronger model, e.g. review-text features (tf-idf) or class-weighted / ordinal models.
