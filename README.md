# CSE 158/258 — Assignment 2: Predicting Clothing Fit (RentTheRunway)

**Task:** Given a customer and a rented item, predict whether it fits: **small / fit / large** (3-class classification).
**Dataset:** RentTheRunway clothing fit data, ~192k rentals (Misra, Wan & McAuley, RecSys 2018) — see `data/README.md`.
**Headline metric:** Macro-F1 (≈74% of rentals are "fit", so accuracy alone rewards always predicting "fit"). Also report accuracy and per-class F1.
**Team:** Ansh, Chloe, Harshitha, Vanshika

Deliverables (due Tuesday of week 10 — check Gradescope for the exact time):
- `workbook.html` — `notebooks/workbook.ipynb`, fully run, exported to HTML
- `video_url.txt` — Google Drive ("anyone with the link") or YouTube link to a ~20 min mp4

## Who does what

| | Owns (code) | Notebook | Presents (video) |
|---|---|---|---|
| **Ansh** | `src/a2/data.py` — cleaning, feature parsing, the shared split | `notebooks/ansh_eda.ipynb` | §2 Data & EDA: source, cleaning, missing values, label imbalance, fit vs. size/category/body type, sparsity |
| **Chloe** | `src/a2/evaluate.py` + baselines and course models in `src/a2/models.py` | `notebooks/chloe_baselines.ipynb` | §3–4 (part 1): majority / item / user baselines, logistic regression on features, latent "true size" model; why Macro-F1 |
| **Harshitha** | Stronger model in `src/a2/models.py` | `notebooks/harshitha_model.ipynb` | §3–4 (part 2): review-text (TF-IDF) model, class imbalance handling, final test-set comparison, error analysis |
| **Vanshika** | Repo, PR reviews, final `workbook.ipynb`, export | `notebooks/workbook.ipynb` | §1 Task framing + §5 Related work; intro/outro; slides and video editing |

Rough video split (~20 min): Vanshika §1 (3 min) · Ansh §2 (4–5 min) · Chloe §3–4 (5 min) · Harshitha §3–4 (5 min) · Vanshika §5 + wrap-up (3 min).

**Cross-checks:** Chloe reviews Ansh's PRs, Harshitha reviews Chloe's, Ansh reviews Harshitha's, everyone reviews Vanshika's workbook. Everyone should be able to explain every part.

## Setup (once)

```bash
git clone https://github.com/rsm-vasrivastava/cse258-assignment2.git
cd cse258-assignment2
uv sync                 # creates .venv with identical packages for everyone
uv run pytest           # should be all green
```

Download the data into `data/` (see `data/README.md`). In VS Code / Jupyter, select the
kernel at `cse258-assignment2/.venv/bin/python`.

Add a package: `uv add <name>` (commit `pyproject.toml` and `uv.lock` together).

## Layout

```
src/a2/data.py       load_dataset(), clean(), split(), targets(), FIT_LABELS   (Ansh)
src/a2/evaluate.py   accuracy, f1_per_class, macro_f1, report(), results_table() (Chloe)
src/a2/models.py     MajorityClass, ItemMajority, UserMajority, + new models   (Chloe, Harshitha)
notebooks/           one notebook per person + workbook.ipynb (final)
tests/               a quick check for every function in src/
data/                local only, never committed
slides/              slide exports (not the video)
```

Every model has the same interface, so they all drop into one results table:

```python
from a2.data import load_dataset, split, targets
from a2.evaluate import report, results_table
from a2.models import MajorityClass

train, valid, test = split(load_dataset())
m = MajorityClass().fit(train)
results_table([report("Majority", targets(valid), m.predict(valid))])
```

## How we work together

1. **Never edit someone else's notebook.** Notebooks merge badly. Shared logic goes in `src/a2/`.
2. **One branch per piece of work:** `git switch -c <name>/<topic>` (e.g. `ansh/eda`).
3. **Start of each session:** `git switch main && git pull`, then `git switch <branch> && git merge main`.
4. **Push often; open a Pull Request** into `main` when something works; your reviewer (above) reads it before merging.
5. **Before pushing a notebook:** Kernel → Restart & Clear All Output, so diffs stay small.
6. **Tune on `valid`; touch `test` once** — only for the final results table.
7. **Every new function in `src/` gets a test** in `tests/`; run `uv run pytest` before pushing.
8. **Don't commit data, videos, or secrets.**

## Timeline

| When | Goal | Who |
|---|---|---|
| Day 1 | Everyone set up + data downloaded; Ansh confirms `clean()` on real data | all / Ansh |
| Days 2–4 | Parallel work on own branch; first results by day 3 | all |
| Day 4 | Related-work notes + slide template | Vanshika |
| Day 5 | Results frozen; final test-set table; workbook assembled and exported | Chloe, Harshitha → Vanshika |
| Day 6 | Each person records their section | all |
| Day 7 | Video edited, uploaded, `video_url.txt` + `workbook.html` submitted | Vanshika |

## Export the workbook

```bash
uv run jupyter nbconvert --to html --execute notebooks/workbook.ipynb --output-dir . --output workbook
head -c 15 workbook.html    # must print <!DOCTYPE html>
```
