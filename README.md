# CSE 158/258 — Assignment 2

**Task:** _TODO: one sentence, e.g. "Predict a user's star rating for a book from their review history and review text."_
**Dataset:** _TODO: name + source link_
**Team:** _TODO: names_

Deliverables (due Tuesday of week 10 — check Gradescope for the exact time):
- `workbook.html` — `notebooks/workbook.ipynb`, fully run, exported to HTML
- `video_url.txt` — Google Drive ("anyone with the link") or YouTube link to a ~20 min mp4

## Setup (once)

```bash
git clone <REPO URL>
cd assignment2
uv sync                 # creates .venv with identical packages for everyone
```

Then download the data (see `data/README.md`). In VS Code / Jupyter, select the
kernel at `assignment2/.venv/bin/python`.

Run the tests any time: `uv run pytest`. Add a package: `uv add <name>` (commit
`pyproject.toml` and `uv.lock` together).

## Layout

| Path | What | Owner |
|---|---|---|
| `src/a2/data.py` | Load + clean data, the **one** train/valid/test split (fixed seed) | A |
| `src/a2/evaluate.py` | The **one** set of metrics everyone reports | B |
| `src/a2/models.py` | Baselines and models, all with `fit(train)` / `predict(rows)` | B, C |
| `notebooks/A_eda.ipynb` | Section 2: EDA, plots, preprocessing | A |
| `notebooks/B_baselines.ipynb` | Sections 3–4: baselines + course models | B |
| `notebooks/C_model.ipynb` | Sections 3–4: stronger / novel model | C |
| `notebooks/workbook.ipynb` | Final combined notebook (5 sections) → `workbook.html` | D |
| `slides/` | Slide deck exports (not the video) | D |
| `tests/` | Quick checks for every function in `src/` | everyone |

Notebooks import shared code instead of copying it:

```python
from a2.data import load_dataset, split
from a2.evaluate import report
from a2.models import GlobalMean
```

## How we work together

1. **Never edit someone else's notebook.** Notebooks merge badly. Shared logic goes in `src/a2/`.
2. **One branch per person per piece of work:** `git switch -c <name>/<topic>`.
3. **Start of each session:** `git switch main && git pull`, then `git switch <your-branch> && git merge main`.
4. **Push often; open a Pull Request** into `main` when something works. One teammate reads it before merging — everyone must be able to explain every part in the video.
5. **Before pushing a notebook:** Kernel → Restart & Clear All Output, so diffs stay small.
6. **Everyone uses `a2.data.split()` and `a2.evaluate.report()`** so results are comparable.
7. **Don't commit data, videos, or secrets.**

## Timeline

| When | Goal |
|---|---|
| Day 1 | Pick task + dataset; data loader and split merged to `main` |
| Days 2–4 | Parallel work on own section/branch |
| Day 5 | Freeze results; D assembles `workbook.ipynb`, runs top-to-bottom, exports HTML |
| Days 6–7 | Slides, record (~20 min, each person presents their section), upload, submit |

## Export the workbook

```bash
uv run jupyter nbconvert --to html --execute notebooks/workbook.ipynb --output-dir . --output workbook
head -c 15 workbook.html    # must print <!DOCTYPE html>
```
