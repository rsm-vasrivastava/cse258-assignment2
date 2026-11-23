import pytest

from a2.models import GlobalMean, ItemMean, UserMean

TRAIN = [
    {"user_id": "u1", "book_id": "b1", "rating": 5},
    {"user_id": "u1", "book_id": "b2", "rating": 3},
    {"user_id": "u2", "book_id": "b1", "rating": 4},
]


def test_global_mean():
    m = GlobalMean().fit(TRAIN)
    assert m.predict([{}, {}]) == [pytest.approx(4.0)] * 2


def test_global_mean_empty_raises():
    with pytest.raises(ValueError):
        GlobalMean().fit([])


def test_item_and_user_mean_with_unseen_fallback():
    item = ItemMean().fit(TRAIN)
    user = UserMean().fit(TRAIN)
    assert item.predict([{"book_id": "b1"}, {"book_id": "new"}]) == [pytest.approx(4.5), pytest.approx(4.0)]
    assert user.predict([{"user_id": "u1"}, {"user_id": "new"}]) == [pytest.approx(4.0), pytest.approx(4.0)]


def test_shrinkage_pulls_towards_global_mean():
    # b2 has one rating (3); with k=1: (3 + 1*4) / (1 + 1) = 3.5
    assert ItemMean(shrinkage=1).fit(TRAIN).predict([{"book_id": "b2"}]) == [pytest.approx(3.5)]
