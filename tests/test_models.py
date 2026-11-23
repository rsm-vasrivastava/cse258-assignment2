import pytest

from a2.models import ItemMajority, MajorityClass, UserMajority

TRAIN = [
    {"user_id": "u1", "item_id": "i1", "fit": "small"},
    {"user_id": "u1", "item_id": "i1", "fit": "small"},
    {"user_id": "u2", "item_id": "i2", "fit": "fit"},
    {"user_id": "u2", "item_id": "i2", "fit": "fit"},
    {"user_id": "u3", "item_id": "i2", "fit": "fit"},
    {"user_id": "u3", "item_id": "i3", "fit": "large"},
]


def test_majority_class():
    assert MajorityClass().fit(TRAIN).predict([{}, {}]) == ["fit", "fit"]


def test_majority_empty_raises():
    with pytest.raises(ValueError):
        MajorityClass().fit([])


def test_item_majority_with_unseen_fallback():
    m = ItemMajority().fit(TRAIN)
    assert m.predict([{"item_id": "i1"}, {"item_id": "i3"}, {"item_id": "new"}]) == ["small", "large", "fit"]


def test_user_majority_tie_prefers_fit():
    # u3 has one "fit" and one "large" -> tie broken towards "fit"
    assert UserMajority().fit(TRAIN).predict([{"user_id": "u3"}]) == ["fit"]


def test_min_count_falls_back_for_rare_groups():
    m = ItemMajority(min_count=2).fit(TRAIN)  # i3 seen once
    assert m.predict([{"item_id": "i3"}, {"item_id": "i1"}]) == ["fit", "small"]
