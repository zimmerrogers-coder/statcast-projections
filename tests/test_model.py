"""The model on simulated hitters whose true talent is known."""

import numpy as np
import pandas as pd
import pytest

from sample import evaluate, marcel, model

SEASONS, TARGET = [2000, 2001, 2002], 2003


def test_prepare_fills_ages_and_league_shifts():
    history = pd.DataFrame([
        {"batter_id": 9, "season": 2000, "age": 24, "pa": 100, "k": 20, "bb": 10, "hbp": 1,
         "balls_measured": 69, "xwobacon_sum": 25.0},
        {"batter_id": 9, "season": 2002, "age": 26, "pa": 200, "k": 50, "bb": 20, "hbp": 2,
         "balls_measured": 128, "xwobacon_sum": 48.0},
        {"batter_id": 4, "season": 2001, "age": 30, "pa": 300, "k": 60, "bb": 30, "hbp": 3,
         "balls_measured": 207, "xwobacon_sum": 80.0},
    ])
    data = model.prepare(history, SEASONS)
    assert list(data.batter_ids) == [4, 9]
    # hitter 4 only played 2001 at 30, so he is 29, 30, 31 across the window
    assert np.allclose(data.age[0], model.scale_age([29, 30, 31]))
    assert list(data.last_age) == [31, 26]
    assert list(data.row_hitter) == [0, 1, 1] and list(data.row_season) == [1, 0, 2]
    # league shift is measured against the last season, so its own shift is 0
    assert all(offsets[-1] == 0 for offsets in data.offsets.values())


@pytest.fixture(scope="module")
def recovered():
    history, truth, league = model.simulate(n_hitters=150, seed=3)
    data = model.prepare(history, SEASONS)
    _, trace = model.fit(data, draws=400, tune=600)
    projection = model.project(trace, data, league, extra_hitters={9001: 24, 9002: 33})
    league_woba = {s: g["woba_num"].sum() / g["pa"].sum() for s, g in history.groupby("season")}
    truth = truth[truth["season"] == TARGET].set_index("batter_id")["true_woba"]
    table = projection.join(marcel.predictions(history, league_woba, TARGET)).join(truth)
    return {"table": table, "diagnostics": model.diagnostics(trace)}


def test_no_divergences(recovered):
    assert recovered["diagnostics"]["divergences"] == 0


def test_closer_to_true_talent_than_marcel(recovered):
    known = recovered["table"].dropna(subset=["Marcel", "true_woba"])

    def rmse(column):
        return float(np.sqrt(((known[column] - known["true_woba"]) ** 2).mean()))

    assert rmse("woba") < rmse("Marcel") < rmse("last season's wOBA")


def test_talent_interval_holds_the_truth_about_as_often_as_claimed(recovered):
    known = recovered["table"].dropna(subset=["true_woba"])
    inside = (known["woba_lo"] <= known["true_woba"]) & (known["true_woba"] <= known["woba_hi"])
    assert 0.62 <= inside.mean() <= 0.95


def test_season_range_is_wider_than_talent_range(recovered):
    table = recovered["table"]
    assert ((table["season_hi"] - table["season_lo"]) > (table["woba_hi"] - table["woba_lo"])).all()


def test_unseen_hitters_get_a_league_level_projection_with_a_wide_range(recovered):
    table = recovered["table"]
    unseen, seen = table.loc[[9001, 9002]], table[table["in_window"]]
    assert not unseen["in_window"].any()
    assert (unseen["woba"] - seen["woba"].median()).abs().max() < 0.05
    regulars = seen[seen["window_pa"] > 800]
    assert ((unseen["woba_hi"] - unseen["woba_lo"]).min()
            > (regulars["woba_hi"] - regulars["woba_lo"]).mean())


def test_scoring_and_luck_floor():
    rows = pd.DataFrame({"pa": [400, 100], "actual": [0.340, 0.300],
                         evaluate.MODEL: [0.340, 0.320], "Marcel": [0.320, 0.300],
                         "last season's wOBA": [0.400, 0.200], "league average": [0.315, 0.315],
                         "result_lo": [0.30, 0.31], "result_hi": [0.38, 0.40]})
    scores = evaluate.score(rows, per_pa_sd=0.5).set_index("method")
    # the model is exact on the 400-PA hitter, Marcel on the 100-PA one: the model wins
    assert scores.loc[evaluate.MODEL, "rmse"] < scores.loc["Marcel", "rmse"]
    assert scores.loc[evaluate.MODEL, "rmse"] == pytest.approx(np.sqrt(100 * 0.02 ** 2 / 500))
    assert scores.loc[evaluate.MODEL, "coverage_80"] == 0.5
    # luck floor: sqrt of PA-weighted mean of sd^2 / PA = sqrt(2 * 0.25 / 500)
    assert scores.loc["Marcel", "luck_floor"] == pytest.approx(np.sqrt(0.5 / 500))
    assert evaluate.compare(rows, "last season's wOBA", resamples=200)["rmse_gain"] > 0
