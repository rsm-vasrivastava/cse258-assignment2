import numpy as np
import pytest
from scipy.optimize import check_grad

from a2.evaluate import average_auc
from a2.models import (GBMFit, ItemMajority, LatentSize, LogisticFit, MajorityClass, TextFit,
                       UserMajority, _ordinal_loss_and_grad)

TRAIN = [
    {"user_id": "u1", "item_id": "i1", "fit": "small"},
    {"user_id": "u1", "item_id": "i1", "fit": "small"},
    {"user_id": "u2", "item_id": "i2", "fit": "fit"},
    {"user_id": "u2", "item_id": "i2", "fit": "fit"},
    {"user_id": "u3", "item_id": "i2", "fit": "fit"},
    {"user_id": "u3", "item_id": "i3", "fit": "large"},
]


# ---------- baselines ----------

def test_majority_class():
    m = MajorityClass().fit(TRAIN)
    assert m.predict([{}, {}]) == ["fit", "fit"]
    np.testing.assert_allclose(m.predict_proba([{}]), [[2 / 6, 3 / 6, 1 / 6]])


def test_majority_empty_raises():
    with pytest.raises(ValueError):
        MajorityClass().fit([])


def test_item_majority_with_unseen_fallback():
    m = ItemMajority().fit(TRAIN)
    assert m.predict([{"item_id": "i1"}, {"item_id": "i3"}, {"item_id": "new"}]) == ["small", "large", "fit"]


def test_group_proba_is_smoothed_towards_prior_and_sums_to_one():
    m = ItemMajority(smoothing=1.0).fit(TRAIN)
    p = m.predict_proba([{"item_id": "i1"}, {"item_id": "new"}])
    # i1: 2 small of 2, prior small = 2/6 -> (2 + 1/3) / 3
    assert p[0, 0] == pytest.approx((2 + 1 / 3) / 3)
    np.testing.assert_allclose(p[1], [2 / 6, 3 / 6, 1 / 6])
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_user_majority_tie_prefers_fit():
    assert UserMajority().fit(TRAIN).predict([{"user_id": "u3"}]) == ["fit"]


def test_min_count_falls_back_for_rare_groups():
    m = ItemMajority(min_count=2).fit(TRAIN)
    assert m.predict([{"item_id": "i3"}, {"item_id": "i1"}]) == ["fit", "small"]


# ---------- 1-LV latent size model ----------

def test_ordinal_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    n_u, n_p, n = 4, 3, 30
    ui, pi = rng.integers(0, n_u, n), rng.integers(0, n_p, n)
    y = rng.integers(0, 3, n)
    x0 = rng.normal(0, 0.5, n_u + n_p + 2)
    f = lambda x: _ordinal_loss_and_grad(x, ui, pi, y, n_u, n_p, 0.1)[0]
    g = lambda x: _ordinal_loss_and_grad(x, ui, pi, y, n_u, n_p, 0.1)[1]
    assert check_grad(f, g, x0) < 1e-5


def _synthetic_sizes(n=3000, seed=0):
    """Customers and products with true sizes; fit depends on their difference."""
    rng = np.random.default_rng(seed)
    cust = rng.normal(0, 2, 150)
    prod = rng.normal(0, 2, 60)
    rows = []
    for _ in range(n):
        c, p = rng.integers(150), rng.integers(60)
        d = cust[c] - prod[p] + rng.normal(0, 0.5)
        fit = "small" if d > 1.5 else "large" if d < -1.5 else "fit"
        rows.append({"user_id": f"c{c}", "item_id": f"p{p}", "size": 8, "fit": fit})
    return rows


def test_latent_size_recovers_size_structure():
    rows = _synthetic_sizes()
    train, test = rows[:2500], rows[2500:]
    m = LatentSize(reg=0.01).fit(train)
    p = m.predict_proba(test)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-9)
    assert average_auc([r["fit"] for r in test], p) > 0.85
    assert m.theta_[0] < m.theta_[1]


def test_latent_size_unseen_user_and_product_get_zero_offset():
    m = LatentSize().fit(_synthetic_sizes(300))
    p = m.predict_proba([{"user_id": "new", "item_id": "new", "size": 8}])
    assert p.shape == (1, 3) and np.isfinite(p).all()


# ---------- feature-based models ----------

def _profile_rows(n=600, seed=0):
    """Heavier-than-usual renters find things small, lighter ones large."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        w = rng.normal(140, 25)
        fit = "small" if w > 165 else "large" if w < 115 else "fit"
        rows.append({"user_id": f"u{i}", "item_id": f"i{i % 20}", "size": 8, "fit": fit,
                     "weight_lbs": w, "height_in": 65.0, "bust_band": 34, "bust_cup": "c", "age": 30,
                     "category": "dress", "body type": "athletic", "rented for": "party"})
    return rows


@pytest.mark.parametrize("model", [LogisticFit(profile_only=True), LogisticFit(profile_only=False),
                                   GBMFit(max_iter=50)])
def test_feature_models_learn_signal_with_correct_column_order(model):
    rows = _profile_rows()
    train, test = rows[:500], rows[500:]
    p = model.fit(train).predict_proba(test)
    assert p.shape == (100, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-9)
    assert average_auc([r["fit"] for r in test], p) > 0.9
    heavy = int(np.argmax([r["weight_lbs"] for r in test]))
    assert np.argmax(p[heavy]) == 0  # column 0 is "small"


def test_text_model_uses_review_words():
    words = {"small": "way too tight and snug", "fit": "true to size perfect", "large": "very loose and baggy"}
    rows = [{"fit": f, "review_text": words[f], "review_summary": ""} for f in ["small", "fit", "large"] * 30]
    m = TextFit(min_df=1).fit(rows)
    pred = m.predict([{"review_text": "it was tight", "review_summary": ""},
                      {"review_text": "so baggy", "review_summary": ""}])
    assert pred == ["small", "large"]
