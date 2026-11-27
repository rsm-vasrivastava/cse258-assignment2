import math

import numpy as np
import pytest

from a2.features import FeatureBuilder, cup_to_number


def row(user, item, size, fit, **kw):
    base = dict(user_id=user, item_id=item, size=size, fit=fit, weight_lbs=130.0, height_in=65.0,
                bust_band=34, bust_cup="c", age=30, category="dress", **{"body type": "athletic",
                "rented for": "party"})
    base.update(kw)
    return base


@pytest.mark.parametrize("cup,expected", [("aa", 0), ("c", 3), ("dd", 5), ("d+", 5), ("ddd/e", 6), ("j", 11), (None, None), ("zz", None)])
def test_cup_to_number(cup, expected):
    assert cup_to_number(cup) == expected


def test_profile_features_bmi_missing_flags_and_unseen_category():
    train = [row("u1", "i1", 8, "fit"), row("u2", "i1", 8, "fit", weight_lbs=None, category="gown"),
             row("u3", "i1", 8, "fit")]
    fb = FeatureBuilder(min_category_count=2)
    X = fb.fit_transform(train)
    assert X.loc[0, "bmi"] == pytest.approx(703 * 130 / 65 ** 2)
    assert X.loc[1, "weight_missing"] == 1 and math.isnan(X.loc[1, "bmi"])
    # "gown" appears once (< min_category_count) -> bucketed as "other"
    assert X.loc[1, "category=other"] == 1 and X.loc[0, "category=dress"] == 1
    Xn = fb.transform([row("u9", "i9", 4, "fit", category="never-seen")])
    assert Xn.loc[0, "category=other"] == 1
    assert list(Xn.columns) == list(X.columns)


def test_history_is_out_of_fold_for_training_rows():
    # every user has exactly one rental, so OOF user history must be empty (NaN / 0 count)
    train = [row(f"u{i}", "i1", 8, "small" if i % 2 else "fit") for i in range(20)]
    fb = FeatureBuilder(n_folds=5)
    X = fb.fit_transform(train)
    assert (X["user_n"] == 0).all()
    assert X["user_p_small"].isna().all()
    # item history comes from the other folds only: 16 rows, never including the row itself
    assert (X["item_n"] == 16).all()


def test_history_for_new_rows_uses_all_training_rows_with_smoothing():
    train = [row("u1", "i1", 8, "small"), row("u1", "i1", 8, "small"), row("u2", "i1", 8, "fit"),
             row("u3", "i2", 8, "large")]
    fb = FeatureBuilder(n_folds=2, smoothing=2.0)
    fb.fit_transform(train)
    X = fb.transform([row("u1", "i1", 10, "fit"), row("new", "new", 8, "fit")])
    prior_small = 2 / 4
    assert X.loc[0, "item_n"] == 3
    assert X.loc[0, "item_p_small"] == pytest.approx((2 + 2.0 * prior_small) / (3 + 2.0))
    assert X.loc[0, "user_p_small"] == pytest.approx((2 + 2.0 * prior_small) / (2 + 2.0))
    # product = (item, size): i1 was only ever rented in size 8, so size 10 is a new product
    assert X.loc[0, "prod_n"] == 0 and math.isnan(X.loc[0, "prod_p_small"])
    # rented size 10 vs. user's usual 8 -> +2
    assert X.loc[0, "size_vs_user"] == pytest.approx(2.0)
    assert X.loc[1, "item_n"] == 0 and math.isnan(X.loc[1, "size_vs_user"])


def test_body_vs_product_fit_renters():
    # people for whom (i1, 8) fit weighed 120 and 140 -> mean 130; a 150 lb renter is +20
    train = [row("a", "i1", 8, "fit", weight_lbs=120.0), row("b", "i1", 8, "fit", weight_lbs=140.0),
             row("c", "i1", 8, "small", weight_lbs=200.0)]
    fb = FeatureBuilder(n_folds=2)
    fb.fit_transform(train)
    X = fb.transform([row("d", "i1", 8, "fit", weight_lbs=150.0)])
    assert X.loc[0, "weight_vs_prod_fit"] == pytest.approx(20.0)


def test_profile_only_columns_exclude_history():
    fb = FeatureBuilder()
    fb.fit_transform([row("u1", "i1", 8, "fit"), row("u2", "i1", 8, "small")])
    assert "bmi" in fb.profile_columns_
    assert not any(c.startswith(("item_", "user_", "prod_")) for c in fb.profile_columns_)


def test_empty_training_set_raises():
    with pytest.raises(ValueError):
        FeatureBuilder().fit_transform([])


def test_output_is_numeric():
    X = FeatureBuilder().fit_transform([row("u1", "i1", 8, "fit"), row("u2", "i2", 4, "large")])
    assert all(np.issubdtype(t, np.number) for t in X.dtypes)
