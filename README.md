# Statcast projection sample

A hierarchical Bayesian model (PyMC) that projects next-season wOBA from public
Statcast data, with an 80% range for every hitter, tested over three holdout
seasons against Marcel.

**Result.** As accurate as Marcel (RMSE .0337 against .0338 over 2024-2026,
1,235 hitter-seasons), with better ordering of hitters, less bias, and ranges
that held 79% of actual results against a target of 80%.

- [reports/model_card.md](reports/model_card.md): data, method, fit quality, results, limits
- [reports/nationals_2027.md](reports/nationals_2027.md): 2027 projections for Washington's 40-man hitters

![Nationals 2027 projections](reports/nationals_2027.png)

## How it works

1. **Download.** Six seasons of Statcast pitch data (2021-2026) from Baseball
   Savant, keeping the pitch that ended each plate appearance: 1.1 million rows
   in PostgreSQL.
2. **Validate.** Internal identities for every hitter-season, and a comparison
   of 120 hitters with Savant's published leaderboard (plate appearances match
   exactly).
3. **Model.** A plate appearance is strikeout, then walk, then hit-by-pitch,
   then a ball in play worth the hitter's contact quality. Each skill has a
   per-hitter level, a yearly drift whose size is learned, and a shared age curve.
4. **Test.** Predict 2024, 2025 and 2026, each from the three seasons before,
   and score against Marcel, last season's wOBA and league average.
5. **Project.** Fit on 2024-2026 and project 2027 for the current 40-man roster.

## Run it

Needs Python 3.11+, and PostgreSQL with a database and role the code can use
(see `src/sample/db.py`; the password is read from `pgpass.conf`).

```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .

.venv\Scripts\python.exe -m sample download     about an hour, resumable
.venv\Scripts\python.exe -m sample build        season totals and validation
.venv\Scripts\python.exe -m sample evaluate     holdout test, about 15 minutes
.venv\Scripts\python.exe -m sample project      2027 projections, about 7 minutes
.venv\Scripts\python.exe -m pytest              17 tests
```

## Layout

```
sql/                 table definitions (mlb schema)
src/sample/
  download.py        Statcast download, one row per plate appearance
  aggregate.py       season totals, league numbers, validation
  marcel.py          Marcel and the other baselines
  model.py           the PyMC model, projection, simulated hitters
  evaluate.py        holdout test and scoring
  project.py         roster and the 2027 table
tests/               Marcel by hand, parsing, model recovery, scoring
reports/             model card, projection table, charts, saved results
```

Data comes from Baseball Savant and MLB's public Stats API. No data is stored
in this repository.
