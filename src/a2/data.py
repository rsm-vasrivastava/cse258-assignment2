"""Loading, cleaning, and the single shared train/valid/test split.

Dataset: RentTheRunway clothing fit (Misra, Wan & McAuley, RecSys 2018),
https://cseweb.ucsd.edu/~jmcauley/datasets.html  ->  renttherunway_final_data.json.gz

Owner: A. Everyone else imports from here; do not re-split data in notebooks.
"""
import ast
import gzip
import json
import random
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

DATA_FILE = "renttherunway_final_data.json.gz"
USER_KEY = "user_id"
ITEM_KEY = "item_id"
TARGET_KEY = "fit"
FIT_LABELS = ["small", "fit", "large"]  # ordered: too small < fits < too large

SEED = 0
AGE_RANGE = (14, 90)  # outside this is a data-entry error (ages 0-9 and 91-117 occur)


def _parse_line(line):
    """Parse one record: strict JSON first, then Python-dict style (McAuley's .gz files)."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def load_raw(filename=DATA_FILE, limit=None):
    """Read a JSON-lines file (optionally .gz) from data/ into a list of raw dicts."""
    path = DATA_DIR / filename
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_parse_line(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


# ---------- field parsers: raw strings -> numbers (None when missing/unparseable) ----------

def parse_weight(s):
    """'137lbs' -> 137.0"""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*lbs?\s*", s or "")
    return float(m.group(1)) if m else None


def parse_height(s):
    """5' 8\" -> 68.0 (inches)"""
    m = re.fullmatch(r"\s*(\d+)\s*'\s*(\d+(?:\.\d+)?)?\s*\"?\s*", s or "")
    if not m:
        return None
    return float(m.group(1)) * 12 + float(m.group(2) or 0)


def parse_bust(s):
    """'34d' -> (34, 'd'); 'd+'/'ddd/e' cups are kept as strings. Missing -> (None, None)."""
    m = re.fullmatch(r"\s*(\d+)\s*([a-z+/]*)\s*", (s or "").lower())
    if not m:
        return None, None
    return int(m.group(1)), (m.group(2) or None)


def parse_number(s, cast=float):
    try:
        return cast(s)
    except (TypeError, ValueError):
        return None


def parse_date(s):
    """'April 20, 2016' -> datetime"""
    try:
        return datetime.strptime(s, "%B %d, %Y")
    except (TypeError, ValueError):
        return None


def clean(row):
    """Return a new dict with numeric fields parsed; the original keys are kept."""
    out = dict(row)
    band, cup = parse_bust(row.get("bust size"))
    age = parse_number(row.get("age"), int)
    if age is not None and not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
        age = None
    out.update(
        weight_lbs=parse_weight(row.get("weight")),
        height_in=parse_height(row.get("height")),
        bust_band=band,
        bust_cup=cup,
        age=age,
        size=parse_number(row.get("size"), int),
        rating=parse_number(row.get("rating"), int),  # 2..10 in steps of 2
        review_date=parse_date(row.get("review_date")),
    )
    return out


def load_dataset(filename=DATA_FILE, limit=None):
    """Load and clean; drops rows without a valid fit label."""
    return [clean(r) for r in load_raw(filename, limit) if r.get(TARGET_KEY) in FIT_LABELS]


def split(rows, valid_frac=0.1, test_frac=0.1, seed=SEED):
    """Shuffle with a fixed seed and return (train, valid, test).

    Everyone must use this so all reported numbers are on the same test set.
    """
    if not 0 <= valid_frac < 1 or not 0 <= test_frac < 1 or valid_frac + test_frac >= 1:
        raise ValueError("valid_frac and test_frac must be in [0, 1) and sum to < 1")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_valid = int(n * valid_frac)
    test = shuffled[:n_test]
    valid = shuffled[n_test:n_test + n_valid]
    train = shuffled[n_test + n_valid:]
    return train, valid, test


def targets(rows, key=TARGET_KEY):
    """Pull the label column out of a list of rows."""
    return [r[key] for r in rows]
