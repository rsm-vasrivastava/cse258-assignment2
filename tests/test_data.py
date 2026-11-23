import gzip
import json
from datetime import datetime

import pytest

from a2 import data

RAW = {
    "fit": "fit", "user_id": "420272", "bust size": "34d", "item_id": "2260466",
    "weight": "137lbs", "rating": "10", "rented for": "vacation",
    "review_text": "An adorable romper!", "body type": "hourglass",
    "review_summary": "So many compliments!", "category": "romper",
    "height": "5' 8\"", "size": 14, "age": "28", "review_date": "April 20, 2016",
}


def _rows(n):
    return [{"user_id": f"u{i % 5}", "item_id": f"i{i % 7}", "fit": "fit"} for i in range(n)]


# ---------- parsers ----------

def test_clean_parses_all_numeric_fields_and_keeps_raw():
    row = data.clean(RAW)
    assert row["weight_lbs"] == 137.0
    assert row["height_in"] == 68.0
    assert (row["bust_band"], row["bust_cup"]) == (34, "d")
    assert (row["age"], row["size"], row["rating"]) == (28, 14, 10)
    assert row["review_date"] == datetime(2016, 4, 20)
    assert row["weight"] == "137lbs"  # raw value untouched
    assert RAW.get("weight_lbs") is None  # input not mutated


def test_clean_missing_fields_become_none():
    row = data.clean({"fit": "small"})
    for key in ["weight_lbs", "height_in", "bust_band", "bust_cup", "age", "size", "rating", "review_date"]:
        assert row[key] is None


@pytest.mark.parametrize("s,expected", [("5' 0\"", 60.0), ("6'", 72.0), ("bad", None), (None, None)])
def test_parse_height(s, expected):
    assert data.parse_height(s) == expected


@pytest.mark.parametrize("s,expected", [("34ddd/e", (34, "ddd/e")), ("32d+", (32, "d+")), ("", (None, None))])
def test_parse_bust_odd_cups(s, expected):
    assert data.parse_bust(s) == expected


# ---------- loading ----------

def test_load_dataset_cleans_and_drops_bad_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    with gzip.open(tmp_path / "r.json.gz", "wt") as f:
        f.write(json.dumps(RAW) + "\n\n")
        f.write(json.dumps({**RAW, "fit": None}) + "\n")
        f.write(str({**RAW, "fit": "large"}) + "\n")  # Python-dict style line
    rows = data.load_dataset("r.json.gz")
    assert [r["fit"] for r in rows] == ["fit", "large"]
    assert rows[0]["height_in"] == 68.0
    assert len(data.load_raw("r.json.gz", limit=1)) == 1


# ---------- split ----------

def test_split_sizes_and_no_overlap():
    rows = _rows(100)
    train, valid, test = data.split(rows, valid_frac=0.1, test_frac=0.2)
    assert (len(train), len(valid), len(test)) == (70, 10, 20)
    ids = lambda part: {id(r) for r in part}
    assert not ids(train) & ids(valid) and not ids(train) & ids(test) and not ids(valid) & ids(test)


def test_split_is_reproducible_and_does_not_mutate_input():
    rows = _rows(50)
    original = list(rows)
    assert data.split(rows) == data.split(rows)
    assert rows == original


def test_split_empty_input():
    assert data.split([]) == ([], [], [])


@pytest.mark.parametrize("valid_frac,test_frac", [(0.5, 0.5), (-0.1, 0.1), (0.1, 1.0)])
def test_split_rejects_bad_fractions(valid_frac, test_frac):
    with pytest.raises(ValueError):
        data.split(_rows(10), valid_frac=valid_frac, test_frac=test_frac)
