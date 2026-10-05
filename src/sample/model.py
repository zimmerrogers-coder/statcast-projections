"""The projection model: a hierarchical Bayesian model of hitter skill over seasons.

A plate appearance is a sequence of steps, each with its own skill:

    strikeout? --no--> walk? --no--> hit by pitch? --no--> ball in play
       p_k              q_bb            q_hbp               worth `contact`

For strikeouts, walks and contact, hitter i in season t has

    skill[i, t] = level[i] + aging[i, t] + drift[i, t] + season[t]

    level[i]     ~ Normal(mu + age_curve(age in first season), sigma)
    aging[i, t]  = age_curve(age[i, t]) - age_curve(age in first season)
    drift[i, t]  = drift[i, t-1] + Normal(0, tau)     starts at 0
    age_curve    = b1 * a + b2 * a^2,  a = (age - 28) / 5
    season[t]    = the league's own shift that season, taken from the data

sigma (how much hitters differ), tau (how much one hitter changes in a year)
and the age curve are estimated from all hitters together. sigma decides how
far a small sample is pulled toward the league. tau decides how much last year
counts against the years before it: it is the learned counterpart of Marcel's
fixed 5/4/3 weights.

Next season's skill carries the drift one step further and moves age up a
year. Putting the steps back together gives wOBA.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

INTERVAL = (10, 90)            # percentiles of an 80% interval
AGE_CENTRE, AGE_SCALE = 28.0, 5.0
SKILLS = ("k", "bb", "c")      # strikeout, walk, contact


@dataclass
class League:
    """Numbers that are the same for every hitter, taken from the last training season."""
    walk_value: float
    hbp_value: float
    unmeasured_share: float     # share of balls in play with no expected wOBA
    unmeasured_value: float     # what those were worth
    calibration: float          # actual value of measured balls / their expected value
    in_play_sd: float           # spread of one ball in play's value

    @classmethod
    def from_row(cls, row) -> "League":
        return cls(float(row["walk_value"]), float(row["hbp_value"]),
                   float(row["unmeasured_share"]), float(row["unmeasured_value"]),
                   float(row["wobacon_measured"]) / float(row["xwobacon"]),
                   float(row["in_play_sd"]))


@dataclass
class ModelData:
    batter_ids: np.ndarray          # hitters, in a fixed order
    seasons: list                   # training seasons, in order
    age: np.ndarray                 # (hitters, seasons), scaled; filled in for missing seasons
    row_hitter: np.ndarray          # for each observed hitter-season: which hitter
    row_season: np.ndarray          #                                  which season
    pa: np.ndarray
    k: np.ndarray
    bb: np.ndarray
    hbp: np.ndarray
    balls: np.ndarray               # measured balls in play
    ball_sum: np.ndarray            # their summed expected wOBA
    last_age: np.ndarray            # each hitter's actual age in the last training season
    offsets: dict                   # skill -> league shift per season, relative to the last one

    @property
    def n_hitters(self) -> int:
        return len(self.batter_ids)


def scale_age(age):
    return (np.asarray(age, dtype=float) - AGE_CENTRE) / AGE_SCALE


def prepare(history: pd.DataFrame, seasons: list) -> ModelData:
    """`history`: one row per hitter-season with batter_id, season, age, pa, k,
    bb, hbp, balls_measured, xwobacon_sum. Only rows in `seasons` are used."""
    rows = history[history["season"].isin(seasons)].sort_values(["batter_id", "season"])
    ids = np.array(sorted(rows["batter_id"].unique()))
    hitter_of = {b: i for i, b in enumerate(ids)}
    season_of = {s: t for t, s in enumerate(seasons)}

    # age in the last training season, worked out from any season he played
    known = rows.assign(age_last=rows["age"] + (seasons[-1] - rows["season"]))
    last_age = known.groupby("batter_id")["age_last"].max().reindex(ids).to_numpy(dtype=float)
    offsets = np.array([s - seasons[-1] for s in seasons], dtype=float)

    # How the league as a whole moved each season, on each skill's own scale,
    # measured against the last training season. Taken straight from the data.
    by_season = rows.groupby("season")[["pa", "k", "bb", "balls_measured", "xwobacon_sum"]].sum()
    by_season = by_season.reindex(seasons)
    league_level = {
        "k": np.log(by_season["k"] / (by_season["pa"] - by_season["k"])),
        "bb": np.log(by_season["bb"] / (by_season["pa"] - by_season["k"] - by_season["bb"])),
        "c": np.log(by_season["xwobacon_sum"] / by_season["balls_measured"]),
    }
    season_offsets = {name: (level - level.iloc[-1]).to_numpy(dtype=float)
                      for name, level in league_level.items()}

    return ModelData(
        batter_ids=ids, seasons=list(seasons), offsets=season_offsets,
        age=scale_age(last_age[:, None] + offsets[None, :]),
        row_hitter=rows["batter_id"].map(hitter_of).to_numpy(dtype=int),
        row_season=rows["season"].map(season_of).to_numpy(dtype=int),
        pa=rows["pa"].to_numpy(dtype=int), k=rows["k"].to_numpy(dtype=int),
        bb=rows["bb"].to_numpy(dtype=int), hbp=rows["hbp"].to_numpy(dtype=int),
        balls=rows["balls_measured"].to_numpy(dtype=int),
        ball_sum=rows["xwobacon_sum"].to_numpy(dtype=float),
        last_age=last_age,
    )


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def build_model(data: ModelData) -> pm.Model:
    n_seasons = len(data.seasons)
    coords = {"hitter": data.batter_ids, "season": data.seasons, "step": data.seasons[1:]}
    h, s = data.row_hitter, data.row_season

    with pm.Model(coords=coords) as model:
        def curve(b1, b2, age):
            return b1 * age + b2 * age ** 2

        def skill(name, centre, centre_sd, sigma_sd, tau_sd):
            mu = pm.Normal(f"mu_{name}", centre, centre_sd)
            b1 = pm.Normal(f"age1_{name}", 0.0, 0.5)
            b2 = pm.Normal(f"age2_{name}", 0.0, 0.5)
            sigma = pm.HalfNormal(f"sigma_{name}", sigma_sd)
            # Gamma(2, .) puts no weight on exactly zero. Hitters do change from
            # year to year, so "no change at all" is not a live possibility, and
            # ruling it out removes a region the sampler gets stuck in.
            tau = pm.Gamma(f"tau_{name}", alpha=2.0, beta=2.0 / tau_sd)
            # A hitter's level is drawn around the league centre for his age.
            # Written this way ("centred") the league numbers sit in the prior
            # of the levels, not beside them in the likelihood. With hundreds
            # of plate appearances per hitter, that is what lets the sampler
            # move: otherwise "the league is higher" and "every hitter is
            # higher" explain the data equally well and it crawls between them.
            first_age = data.age[:, 0]
            level = pm.Normal(f"level_{name}", mu + curve(b1, b2, first_age), sigma, dims="hitter")
            # The yearly steps stay as tau * e: one season says little about a
            # single step, and for weakly-informed values that form is better.
            e = pm.Normal(f"e_{name}", 0.0, 1.0, dims=("hitter", "step"))
            drift = pt.concatenate([pt.zeros((data.n_hitters, 1)), pt.cumsum(tau * e, axis=1)],
                                   axis=1)
            aging = curve(b1, b2, data.age) - curve(b1, b2, first_age)[:, None]
            return level[:, None] + aging + drift + data.offsets[name][None, :]

        # last argument: prior mean of the yearly step size
        eta_k = skill("k", _logit(0.22), 1.0, 1.0, 0.15)
        eta_bb = skill("bb", _logit(0.10), 1.0, 1.0, 0.15)
        eta_c = skill("c", float(np.log(0.37)), 0.5, 0.5, 0.05)

        pm.Binomial("k_obs", n=data.pa, p=pm.math.invlogit(eta_k[h, s]), observed=data.k)
        pm.Binomial("bb_obs", n=data.pa - data.k, p=pm.math.invlogit(eta_bb[h, s]),
                    observed=data.bb)

        # Hit by pitch is a real skill too (some hitters crowd the plate), but a
        # steadier one: a talent per hitter, with no drift and no age curve.
        league_hbp = pm.Normal("league_hbp", _logit(0.015), 1.0)
        sigma_hbp = pm.HalfNormal("sigma_hbp", 1.0)
        z_hbp = pm.Normal("z_hbp", 0.0, 1.0, dims="hitter")      # rare event: non-centred
        pm.Binomial("hbp_obs", n=data.pa - data.k - data.bb,
                    p=pm.math.invlogit(league_hbp + sigma_hbp * z_hbp[h]), observed=data.hbp)

        # Contact: one ball's expected wOBA is positive and lopsided, so Gamma
        # with mean `contact`. A season's SUM over n balls is then exactly
        # Gamma with shape n * alpha, which lets the model use season totals.
        has_balls = data.balls > 0
        alpha = pm.Gamma("alpha", 2.0, 1.0)
        contact = pm.math.exp(eta_c[h, s])[has_balls]
        pm.Gamma("ball_obs", alpha=alpha * data.balls[has_balls], beta=alpha / contact,
                 observed=np.maximum(data.ball_sum[has_balls], 1e-6))
    return model


def fit(data: ModelData, draws: int = 1000, tune: int = 1000, chains: int = 4,
        seed: int = 2027, progressbar: bool = False):
    model = build_model(data)
    with model:
        trace = pm.sample(draws, tune=tune, chains=chains, target_accept=0.9,
                          random_seed=seed, progressbar=progressbar)
    return model, trace


def draws_of(trace, name: str) -> np.ndarray:
    values = np.asarray(trace.posterior[name].values)
    return values.reshape(-1, *values.shape[2:])


def diagnostics(trace) -> dict:
    """divergences must be 0; max_rhat under 1.01; min_ess comfortably in the hundreds."""
    import arviz as az

    names = [n for n in trace.posterior.data_vars if not n.startswith(("z_", "e_", "level_"))]
    rhat, ess = az.rhat(trace, var_names=names), az.ess(trace, var_names=names)

    def extreme(result, fn):
        dataset = result.to_dataset() if hasattr(result, "to_dataset") else result
        return float(fn([float(fn(np.asarray(dataset[v].values))) for v in dataset.data_vars]))

    return {"divergences": int(np.asarray(trace.sample_stats["diverging"].values).sum()),
            "max_rhat": extreme(rhat, np.nanmax), "min_ess": extreme(ess, np.nanmin),
            "draws": int(draws_of(trace, "alpha").shape[0])}


def league_summary(trace) -> pd.DataFrame:
    rows = []
    for name in trace.posterior.data_vars:
        if name.startswith(("z_", "e_", "level_")):
            continue
        values = draws_of(trace, name)
        values = values.reshape(len(values), -1)
        for column in range(values.shape[1]):
            label = name if values.shape[1] == 1 else f"{name}[{column}]"
            low, high = np.percentile(values[:, column], INTERVAL)
            rows.append({"parameter": label, "mean": float(values[:, column].mean()),
                         "lo_80": float(low), "hi_80": float(high)})
    return pd.DataFrame(rows)


def _inverse_logit(x):
    return 1 / (1 + np.exp(-x))


def project(trace, data: ModelData, league: League, full_season_pa: int = 500,
            result_pa: dict | None = None, extra_hitters: dict | None = None,
            n_sim: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Next season for every hitter in the model.

    For each kept posterior draw: carry each hitter's drift one more season,
    add a year of age, and rebuild wOBA. That gives a spread of plausible true
    talents. The spread of the wOBA he would actually post over `m` plate
    appearances is wider, because m trips to the plate carry their own luck.

    result_pa:      {batter_id: m} to get `result_lo/hi` at each hitter's own m
                    (used when scoring against a real season)
    extra_hitters:  {batter_id: age next season} for hitters with no history in
                    the window; they are drawn fresh from the league distributions
    """
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(draws_of(trace, "alpha")), size=n_sim, replace=False)
    extra = extra_hitters or {}
    ids = np.concatenate([data.batter_ids, np.array(list(extra), dtype=data.batter_ids.dtype)])
    n_known, n_all = data.n_hitters, len(ids)
    next_age = scale_age(np.concatenate([data.last_age + 1, np.array(list(extra.values()),
                                                                      dtype=float)]))

    first_age = np.concatenate([data.age[:, 0], next_age[n_known:]])

    def next_skill(name):
        mu = draws_of(trace, f"mu_{name}")[keep][:, None]
        b1 = draws_of(trace, f"age1_{name}")[keep][:, None]
        b2 = draws_of(trace, f"age2_{name}")[keep][:, None]
        sigma = draws_of(trace, f"sigma_{name}")[keep][:, None]
        tau = draws_of(trace, f"tau_{name}")[keep][:, None]

        def curve(age):
            return b1 * age[None, :] + b2 * age[None, :] ** 2

        level = np.empty((n_sim, n_all))
        level[:, :n_known] = draws_of(trace, f"level_{name}")[keep]
        # a hitter never seen: a level drawn from the population at his age
        # (his "first season" is next season, so no aging or drift is added)
        level[:, n_known:] = (mu + curve(first_age)[:, n_known:]
                              + sigma * rng.standard_normal((n_sim, n_all - n_known)))
        drift = np.zeros((n_sim, n_all))
        drift[:, :n_known] = tau * draws_of(trace, f"e_{name}")[keep].sum(axis=2)
        step = tau * rng.standard_normal((n_sim, n_all))
        step[:, n_known:] = 0.0
        return level + curve(next_age) - curve(first_age) + drift + step

    p_k = _inverse_logit(next_skill("k"))
    q_bb = _inverse_logit(next_skill("bb"))
    contact = np.exp(next_skill("c"))
    z_hbp = np.concatenate([draws_of(trace, "z_hbp")[keep],
                            rng.standard_normal((n_sim, n_all - n_known))], axis=1)
    q_hbp = _inverse_logit(draws_of(trace, "league_hbp")[keep][:, None]
                           + draws_of(trace, "sigma_hbp")[keep][:, None] * z_hbp)

    in_play_value = ((1 - league.unmeasured_share) * contact * league.calibration
                     + league.unmeasured_share * league.unmeasured_value)
    reach = 1 - p_k
    p_bb = reach * q_bb
    p_hbp = reach * (1 - q_bb) * q_hbp
    p_in_play = reach * (1 - q_bb) * (1 - q_hbp)
    talent_woba = (p_bb * league.walk_value + p_hbp * league.hbp_value
                   + p_in_play * in_play_value)

    def simulate_result(m: np.ndarray) -> np.ndarray:
        """wOBA actually posted over m plate appearances, one value per draw and hitter."""
        strikeouts = rng.binomial(m, p_k)
        walks = rng.binomial(m - strikeouts, q_bb)
        hit = rng.binomial(m - strikeouts - walks, q_hbp)
        balls = m - strikeouts - walks - hit
        on_contact = rng.normal(balls * in_play_value, league.in_play_sd * np.sqrt(balls))
        return (walks * league.walk_value + hit * league.hbp_value + on_contact) / m

    out = pd.DataFrame({"batter_id": ids, "in_window": np.arange(n_all) < n_known})
    for column, values in (("woba", talent_woba), ("k_rate", p_k), ("bb_rate", p_bb),
                           ("contact", contact)):
        low, high = np.percentile(values, INTERVAL, axis=0)
        out[column], out[f"{column}_lo"], out[f"{column}_hi"] = values.mean(axis=0), low, high

    full = simulate_result(np.full(n_all, full_season_pa))
    out["season_lo"], out["season_hi"] = np.percentile(full, INTERVAL, axis=0)
    if result_pa is not None:
        m = np.array([max(int(result_pa.get(b, 0)), 1) for b in ids])
        own = simulate_result(m)
        out["result_lo"], out["result_hi"] = np.percentile(own, INTERVAL, axis=0)
    return out.set_index("batter_id")


def simulate(n_hitters: int = 300, n_seasons: int = 4, seed: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, League]:
    """Invent hitters with known skill paths over several seasons.

    Returns (history, truth, league). `history` looks like mlb.hitter_season.
    `truth` has each hitter's true wOBA in every season. Playing time is tied
    to talent, as it is in real life: better hitters bat more."""
    rng = np.random.default_rng(seed)
    league = League(walk_value=0.69, hbp_value=0.72, unmeasured_share=0.0, unmeasured_value=0.0,
                    calibration=1.0, in_play_sd=0.50)
    t = {"k": (_logit(0.22), 0.45, 0.12, 0.10, 0.10), "bb": (_logit(0.10), 0.35, 0.10, 0.0, -0.05),
         "c": (float(np.log(0.37)), 0.14, 0.05, -0.02, -0.03)}   # centre, sigma, tau, b1, b2
    alpha = 0.9
    age0 = rng.integers(21, 36, n_hitters).astype(float)

    skill = {}
    for name, (centre, sigma, tau, b1, b2) in t.items():
        talent = sigma * rng.standard_normal(n_hitters)
        drift = np.concatenate([np.zeros((n_hitters, 1)),
                                np.cumsum(tau * rng.standard_normal((n_hitters, n_seasons - 1)),
                                          axis=1)], axis=1)
        age = scale_age(age0[:, None] + np.arange(n_seasons)[None, :])
        skill[name] = centre + b1 * age + b2 * age ** 2 + talent[:, None] + drift
    p_k, q_bb, contact = _inverse_logit(skill["k"]), _inverse_logit(skill["bb"]), np.exp(skill["c"])
    q_hbp = _inverse_logit(_logit(0.015) + 0.5 * rng.standard_normal(n_hitters))[:, None]
    reach = 1 - p_k
    woba = (reach * q_bb * league.walk_value + reach * (1 - q_bb) * q_hbp * league.hbp_value
            + reach * (1 - q_bb) * (1 - q_hbp) * contact)

    history, truth = [], []
    for i in range(n_hitters):
        for s in range(n_seasons):
            quality = (woba[i, s] - 0.30) / 0.04                 # better hitters play more
            pa = int(np.clip(rng.normal(250 + 120 * quality, 150), 0, 700))
            truth.append({"batter_id": i, "season": 2000 + s, "true_woba": woba[i, s]})
            if pa < 1:
                continue
            k = rng.binomial(pa, p_k[i, s])
            bb = rng.binomial(pa - k, q_bb[i, s])
            hbp = rng.binomial(pa - k - bb, q_hbp[i, 0])
            balls = pa - k - bb - hbp
            ball_sum = rng.gamma(alpha * balls, contact[i, s] / alpha) if balls else 0.0
            # actual results: expected value plus the luck of each ball
            actual = max(rng.normal(ball_sum, league.in_play_sd * 0.8 * np.sqrt(max(balls, 1))), 0.0)
            history.append({"batter_id": i, "season": 2000 + s, "age": int(age0[i] + s), "pa": pa,
                            "k": k, "bb": bb, "hbp": hbp, "balls_in_play": balls,
                            "balls_measured": balls, "xwobacon_sum": ball_sum,
                            "woba_num": bb * league.walk_value + hbp * league.hbp_value + actual})
    return pd.DataFrame(history), pd.DataFrame(truth), league
