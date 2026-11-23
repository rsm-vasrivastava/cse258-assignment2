# Data

Raw data files live here locally but are **not committed** (see `.gitignore`).

## How to get the data
1. Go to https://cseweb.ucsd.edu/~jmcauley/datasets.html → **Clothing Fit Data** →
   download `renttherunway_final_data.json.gz` (~192k rentals).
2. Put it in this folder unchanged: `data/renttherunway_final_data.json.gz`
   (no need to unzip — `a2.data.load_dataset()` reads the `.gz` directly).

## Fields (raw → cleaned by `a2.data.clean`)
| Raw | Example | Cleaned |
|---|---|---|
| `fit` | `"fit"` / `"small"` / `"large"` | label (target) |
| `user_id`, `item_id` | `"420272"` | unchanged |
| `weight` | `"137lbs"` | `weight_lbs` = 137.0 |
| `height` | `5' 8"` | `height_in` = 68.0 |
| `bust size` | `"34d"` | `bust_band` = 34, `bust_cup` = `"d"` |
| `age`, `size`, `rating` | `"28"`, `14`, `"10"` | ints (`rating` is 2–10) |
| `review_date` | `"April 20, 2016"` | `datetime` |
| `category`, `body type`, `rented for` | `"romper"`, `"hourglass"`, `"vacation"` | unchanged (categorical) |
| `review_text`, `review_summary` | free text | unchanged |

Missing or unparseable values become `None`.
