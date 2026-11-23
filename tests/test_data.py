import gzip
import json

import pytest

from a2 import data


def _rows(n):
    return [{"user_id": f"u{i % 5}", "book_id": f"b{i % 7}", "rating": i % 6} for i in range(n)]


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


def test_load_dataset_json_gz_and_python_dict_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    (tmp_path / "a.json").write_text(json.dumps({"rating": 5}) + "\n\n" + json.dumps({"rating": 3}) + "\n")
    with gzip.open(tmp_path / "b.json.gz", "wt") as f:
        f.write("{'rating': 4, 'ok': True}\n")
    assert data.load_dataset("a.json") == [{"rating": 5}, {"rating": 3}]
    assert data.load_dataset("a.json", limit=1) == [{"rating": 5}]
    assert data.load_dataset("b.json.gz") == [{"rating": 4, "ok": True}]
