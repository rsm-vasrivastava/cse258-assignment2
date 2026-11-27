# CSE 158/258 — Assignment 2: Will It Fit? Predicting Clothing Fit on RentTheRunway

**Task:** Before a customer rents a garment in a given size, predict whether it will be reported **small / fit / large** (3-class classification).
**Dataset:** RentTheRunway clothing fit data, 192,544 rentals (Misra, Wan & McAuley, RecSys 2018). See `data/README.md`.
**Team:** Ansh, Chloe, Harshitha, Vanshika

## Results (held-out test set, 19,254 rentals)

| Model | Avg AUC | Macro-F1 | Accuracy |
|---|---|---|---|
| **Gradient boosting (profile + history)** | **0.775** | **0.567** | 0.699 |
| Logistic regression (profile + history) | 0.764 | 0.563 | 0.710 |
| Item fit history (baseline) | 0.744 | 0.542 | 0.708 |
| 1-LV latent size (Sembium et al. 2017) | 0.711 | 0.513 | 0.670 |
| Logistic regression (profile only) | 0.612 | 0.419 | 0.618 |
| User fit history (baseline) | 0.537 | 0.371 | 0.589 |
| Majority class ("fit") | 0.500 | 0.283 | 0.736 |
| *Review text, post-hoc only (not available before the rental)* | *0.854* | *0.688* | *0.803* |

Full analysis, charts, bootstrap CIs, cold-start breakdown and related work: **`workbook.html`** (rendered) / `notebooks/workbook.ipynb`.

## Deliverables (due Tuesday of week 10; check Gradescope)
- [x] `workbook.html`: `notebooks/workbook.ipynb`, executed and exported (passes the autograder's `<!DOCTYPE html>` check)
- [ ] `video_url.txt`: Google Drive ("Anyone with the link") or YouTube link to the ~20 min mp4. See `slides/presentation_outline.md`

## Who presents what

| | Section | Notebook part |
|---|---|---|
| **Vanshika** | §1 Predictive task · §5 Related work + conclusions | §1, §5 |
| **Ansh** | §2 Data, EDA, preprocessing | §2 |
| **Chloe** | §3–4 part 1: baselines, 1-LV, logistic regression, metrics & decision weights | §3 (1-LV, fitting), §4 tables |
| **Harshitha** | §3–4 part 2: features, gradient boosting, significance, cold start, errors, text | §3 (features), §4 rest |

Talking points per slide: `slides/presentation_outline.md`. Before recording, each person should read their part of `src/a2/` and be able to explain it.

## Setup

```bash
git clone https://github.com/rsm-vasrivastava/cse258-assignment2.git
cd cse258-assignment2
uv sync                 # creates .venv with identical packages for everyone
uv run pytest           # 60 tests, all should pass
```

Download the data into `data/` (see `data/README.md`). In VS Code / Jupyter, select the
kernel at `cse258-assignment2/.venv/bin/python`.

## Layout

```
src/a2/data.py       load + clean (parsers for weight/height/bust/age...), fixed-seed 80/10/10 split
src/a2/features.py   FeatureBuilder: profile features + out-of-fold history features
src/a2/models.py     MajorityClass, ItemMajority, UserMajority, LatentSize (1-LV),
                     LogisticFit, GBMFit, TextFit, all with fit() / predict_proba()
src/a2/evaluate.py   Avg AUC, Macro-F1, per-class F1, decision-weight tuning, paired bootstrap CI
tests/               60 unit tests (incl. finite-difference gradient check for 1-LV)
notebooks/workbook.ipynb   the submission notebook (5 sections)
workbook.html        exported submission
slides/              presentation outline
data/                local only, never committed
```

## Reproduce / re-export

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/workbook.ipynb   # ~4 min
uv run jupyter nbconvert --to html notebooks/workbook.ipynb --output-dir . --output workbook
head -c 15 workbook.html    # must print <!DOCTYPE html>
```

All results are deterministic (fixed seeds), so re-running reproduces the numbers above.

## Working on it
1. Shared logic goes in `src/a2/` with a test in `tests/`; run `uv run pytest` before pushing.
2. Branch per change (`git switch -c <name>/<topic>`), open a PR, and have a teammate review it.
3. Tune on validation; the test set is scored once, in the notebook.
4. Don't commit data, videos or secrets.
