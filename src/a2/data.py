"""Loading, cleaning, and the single shared train/valid/test split.

Owner: A. Everyone else imports from here; do not re-split data in notebooks.
"""
import ast
import gzip
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# TODO(A): set these once the dataset is chosen.
DATA_FILE = "young_adult_20000.json"
USER_KEY = "user_id"
ITEM_KEY = "book_id"
TARGET_KEY = "rating"

SEED = 0


def _parse_line(line):
    """Parse one record: strict JSON first, then Python-dict style (McAuley's .gz files)."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def load_dataset(filename=DATA_FILE, limit=None):
    """Read a JSON-lines file (optionally .gz) from data/ into a list of dicts."""
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
