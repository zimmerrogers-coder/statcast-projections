"""Marcel worked by hand."""

import pandas as pd
import pytest

from sample import marcel

LEAGUE = {2023: 0.320, 2024: 0.310, 2025: 0.315}


def season(batter, year, pa, woba, age):
    return {"batter_id": batter, "season": year, "pa": pa, "woba_num": woba * pa, "age": age}


def test_age_factor():
    assert marcel.age_factor(29) == 1.0
    assert marcel.age_factor(25) == pytest.approx(1.024)      # 4 years under: +0.6% each
    assert marcel.age_factor(33) == pytest.approx(0.988)      # 4 years over:  -0.3% each


def test_three_full_seasons():
    history = pd.DataFrame([season(1, 2023, 600, 0.300, 26), season(1, 2024, 600, 0.330, 27),
                            season(1, 2025, 600, 0.360, 28)])
    out = marcel.predictions(history, LEAGUE, 2026).loc[1]

    # weights 5/4/3 on 2025/2024/2023
    weighted_pa = 5 * 600 + 4 * 600 + 3 * 600                        # 7200
    weighted_num = (5 * 0.360 + 4 * 0.330 + 3 * 0.300) * 600         # 2412
    weighted_league = (5 * 0.315 + 4 * 0.310 + 3 * 0.320) / 12       # 0.31458
    regressed = (weighted_num + 1200 * weighted_league) / (weighted_pa + 1200)
    # he is 29 in 2026: no age adjustment
    assert out["Marcel"] == pytest.approx(regressed)
    assert 0.330 < out["Marcel"] < 0.335       # between his recent form and the league
    assert out["last season's wOBA"] == pytest.approx(0.360)
    assert out["league average"] == 0.315
    assert out["window_pa"] == 1800


def test_small_sample_is_pulled_almost_all_the_way_to_the_league():
    history = pd.DataFrame([season(2, 2025, 20, 0.600, 29)])        # 30 in the target season
    out = marcel.predictions(history, LEAGUE, 2026).loc[2]
    regressed = (5 * 20 * 0.600 + 1200 * 0.315) / (5 * 20 + 1200)   # 0.3369
    assert out["Marcel"] == pytest.approx(regressed * (1 - 0.003))


def test_missing_last_season_uses_what_exists():
    history = pd.DataFrame([season(3, 2023, 500, 0.350, 24)])       # out for two years
    out = marcel.predictions(history, LEAGUE, 2026).loc[3]
    regressed = (3 * 500 * 0.350 + 1200 * 0.320) / (3 * 500 + 1200)
    assert out["Marcel"] == pytest.approx(regressed * marcel.age_factor(27))
    assert out["last season's wOBA"] == pytest.approx(0.350)


def test_seasons_outside_the_window_are_ignored():
    history = pd.DataFrame([season(4, 2021, 600, 0.450, 30), season(4, 2025, 600, 0.300, 34)])
    only_recent = pd.DataFrame([season(4, 2025, 600, 0.300, 34)])
    assert (marcel.predictions(history, LEAGUE, 2026).loc[4, "Marcel"]
            == marcel.predictions(only_recent, LEAGUE, 2026).loc[4, "Marcel"])
    # and a hitter with nothing in the window is not projected at all
    assert 5 not in marcel.predictions(history, LEAGUE, 2026).index
