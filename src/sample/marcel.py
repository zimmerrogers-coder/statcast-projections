"""The baselines the model has to beat.

Marcel is Tom Tango's deliberately simple projection system, "the minimum
level of competence that you should expect from any forecaster". It uses only
the last three seasons:

  1. weight them 5, 4, 3 (most recent first)
  2. regress to the mean by adding 1,200 plate appearances of league average
  3. adjust for age: +0.6% a year under 29, -0.3% a year over

It has no notion of uncertainty and no player-specific shrinkage, yet it is
hard to beat, which is what makes it the right yardstick.
"""

import pandas as pd

WEIGHTS = (5, 4, 3)              # seasons t-1, t-2, t-3
REGRESSION_PA = 1200
PEAK_AGE = 29
YOUNG_GAIN, OLD_LOSS = 0.006, 0.003


def age_factor(age: float) -> float:
    if age < PEAK_AGE:
        return 1 + (PEAK_AGE - age) * YOUNG_GAIN
    return 1 - (age - PEAK_AGE) * OLD_LOSS


def predictions(seasons: pd.DataFrame, league_woba: dict, target: int) -> pd.DataFrame:
    """Marcel and two simpler baselines for every hitter with history in the
    three seasons before `target`.

    seasons:      one row per hitter-season with batter_id, season, age, pa, woba_num
    league_woba:  {season: league wOBA}
    Returns one row per hitter, indexed by batter_id."""
    weight = {target - 1 - i: w for i, w in enumerate(WEIGHTS)}
    window = seasons[seasons["season"].isin(weight)].copy()
    window["w"] = window["season"].map(weight)
    window["w_pa"] = window["w"] * window["pa"]
    window["w_num"] = window["w"] * window["woba_num"]
    window["w_league"] = window["w_pa"] * window["season"].map(league_woba)

    grouped = window.groupby("batter_id")
    total = grouped[["w_pa", "w_num", "w_league"]].sum()
    regressed = ((total["w_num"] + REGRESSION_PA * total["w_league"] / total["w_pa"])
                 / (total["w_pa"] + REGRESSION_PA))

    latest = window.sort_values("season").groupby("batter_id").tail(1).set_index("batter_id")
    age_in_target = latest["age"] + (target - latest["season"])

    out = pd.DataFrame(index=total.index)
    out["Marcel"] = regressed * age_in_target.map(age_factor)
    out["last season's wOBA"] = latest["woba_num"] / latest["pa"]
    out["league average"] = league_woba[target - 1]
    out["window_pa"] = grouped["pa"].sum()
    return out
