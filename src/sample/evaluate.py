"""Holdout test: predict a season from the three before it, and compare with Marcel.

For each target season (2024, 2025, 2026) the model is fitted on the three
previous seasons only. Marcel and the other baselines see the same three
seasons. Everyone then predicts each hitter's wOBA in the target season, and
the predictions are scored against what actually happened.
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from sample import db, marcel, model  # noqa: E402

TARGETS = (2024, 2025, 2026)
MIN_PA = 100
MODEL = "Bayesian model"
BASELINES = ["Marcel", "last season's wOBA", "league average"]
METHODS = [MODEL, *BASELINES]
MAX_RHAT = 1.01

_INK, _SECONDARY, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
_BLUE, _NEUTRAL = "#2a78d6", "#898781"

PROJECTION_COLUMNS = ["version", "batter_id", "in_window", "window_pa", "woba", "woba_lo",
                      "woba_hi", "season_lo", "season_hi", "k_rate", "k_rate_lo", "k_rate_hi",
                      "bb_rate", "bb_rate_lo", "bb_rate_hi", "contact", "contact_lo",
                      "contact_hi", "marcel"]


def load(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = db.read(conn, "SELECT * FROM mlb.hitter_season")
    league = db.read(conn, "SELECT * FROM mlb.league_season").set_index("season")
    return history, league


def fit_and_project(conn, history, league, target: int, version: str, result_pa=None,
                    extra_hitters=None, draws: int = 2500, tune: int = 1500) -> dict:
    """Fit on the three seasons before `target`, project `target`, and store
    the run. Returns the projection table and the fit's diagnostics."""
    seasons = [target - 3, target - 2, target - 1]
    data = model.prepare(history, seasons)
    started = time.perf_counter()
    _, trace = model.fit(data, draws=draws, tune=tune)
    seconds = time.perf_counter() - started
    diagnostics = model.diagnostics(trace)

    projection = model.project(trace, data, model.League.from_row(league.loc[target - 1]),
                               result_pa=result_pa, extra_hitters=extra_hitters)
    baselines = marcel.predictions(history, league["woba"].to_dict(), target)
    projection = projection.join(baselines[["Marcel", "last season's wOBA", "league average",
                                            "window_pa"]])
    projection["league average"] = float(league.loc[target - 1, "woba"])

    conn.execute("DELETE FROM mlb.model_version WHERE version = %s", (version,))
    conn.execute(
        "INSERT INTO mlb.model_version (version, target_season, train_seasons, hitters, pa, "
        "fit_seconds, diagnostics) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (version, target, seasons, data.n_hitters, int(data.pa.sum()), seconds, Jsonb(diagnostics)))
    db.insert(conn, "mlb.model_parameter", model.league_summary(trace).assign(version=version),
              ["version", "parameter", "mean", "lo_80", "hi_80"])
    stored = projection.reset_index().assign(version=version).rename(columns={"Marcel": "marcel"})
    db.insert(conn, "mlb.projection", stored, PROJECTION_COLUMNS)
    conn.commit()
    return {"projection": projection, "diagnostics": diagnostics, "seconds": seconds,
            "hitters": data.n_hitters, "parameters": model.league_summary(trace)}


def _rmse(rows: pd.DataFrame, method: str) -> float:
    return float(np.sqrt(np.average((rows[method] - rows["actual"]) ** 2, weights=rows["pa"])))


def luck_floor(rows: pd.DataFrame, per_pa_sd: float) -> float:
    """The error a perfect forecaster would still make: even with true talent
    known exactly, wOBA over m plate appearances misses it by about
    (spread of one PA) / sqrt(m). Weighted like the scores."""
    return float(np.sqrt(np.average(per_pa_sd ** 2 / rows["pa"], weights=rows["pa"])))


def score(rows: pd.DataFrame, per_pa_sd: float) -> pd.DataFrame:
    floor = luck_floor(rows, per_pa_sd)
    out = []
    for method in METHODS:
        error = rows[method] - rows["actual"]
        entry = {"method": method, "hitters": len(rows), "pa": int(rows["pa"].sum()),
                 "rmse": _rmse(rows, method),
                 "mae": float(np.average(error.abs(), weights=rows["pa"])),
                 "bias": float(np.average(error, weights=rows["pa"])),
                 "rank_corr": (float(spearmanr(rows[method], rows["actual"]).statistic)
                               if rows[method].nunique() > 1 else None),
                 "coverage_80": None, "luck_floor": floor}
        if method == MODEL:
            inside = (rows["result_lo"] <= rows["actual"]) & (rows["actual"] <= rows["result_hi"])
            entry["coverage_80"] = float(inside.mean())
        out.append(entry)
    return pd.DataFrame(out)


def compare(rows: pd.DataFrame, other: str, resamples: int = 4000, seed: int = 0) -> dict:
    """How sure is it that the model beats `other`? Redraw the hitters with
    replacement many times and see how often its error is the smaller one."""
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(resamples):
        sample = rows.iloc[rng.integers(0, len(rows), size=len(rows))]
        gains.append(_rmse(sample, other) - _rmse(sample, MODEL))
    gains = np.array(gains)
    low, high = np.percentile(gains, [10, 90])
    return {"versus": other, "rmse_gain": _rmse(rows, other) - _rmse(rows, MODEL),
            "gain_lo": float(low), "gain_hi": float(high),
            "share_better": float((gains > 0).mean())}


def holdout_rows(conn, history, league, target: int) -> tuple[pd.DataFrame, dict]:
    """One row per scored hitter in `target`: actual wOBA and every prediction."""
    actual = history[(history["season"] == target) & (history["pa"] >= MIN_PA)].set_index("batter_id")
    result = fit_and_project(conn, history, league, target, f"holdout-{target}",
                             result_pa=actual["pa"].to_dict())
    projection = result["projection"]
    rows = pd.DataFrame({"pa": actual["pa"], "actual": actual["woba_num"] / actual["pa"]})
    rows = rows.join(projection[["woba", "result_lo", "result_hi", *BASELINES, "window_pa"]],
                     how="inner").rename(columns={"woba": MODEL})
    rows = rows.dropna(subset=["Marcel"])          # hitters with history in the window
    rows.insert(0, "target", target)
    return rows.reset_index(), result


def draw(scores: pd.DataFrame, path) -> None:
    targets = [t for t in scores["target_season"].unique()]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.6 * len(targets), 3.4),
                             facecolor=_SURFACE, sharey=True)
    order = [m for m in reversed(METHODS)]
    for ax, target in zip(axes, targets):
        part = scores[scores["target_season"] == target].set_index("method").loc[order]
        floor = part["luck_floor"].iloc[0]
        low = min(part["rmse"].min(), floor)
        colours = [_BLUE if m == MODEL else _NEUTRAL for m in part.index]
        ax.axvline(floor, color=_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.hlines(part.index, low - 0.001, part["rmse"], color=_GRID, linewidth=1.5, zorder=1)
        ax.scatter(part["rmse"], part.index, s=60, color=colours, edgecolor=_SURFACE,
                   linewidth=1.5, zorder=3)
        for value, method in zip(part["rmse"], part.index):
            ax.annotate(f"{value:.4f}", (value, method), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=8, color=_INK)
        title = "All three seasons" if target == "all" else f"Predicting {target}"
        ax.set_title(title, loc="left", fontsize=10.5, color=_INK, pad=14)
        ax.text(0, 1.02, f"{part['hitters'].iloc[0]} hitters", transform=ax.transAxes,
                fontsize=8, color=_SECONDARY)
        ax.set_xlim(low - 0.001, part["rmse"].max() + 0.004)
        ax.set_ylim(-0.6, len(order) - 0.3)
        ax.tick_params(axis="x", colors=_MUTED, labelsize=7.5, length=0)
        ax.tick_params(axis="y", colors=_INK, labelsize=9, length=0)
        ax.set_facecolor(_SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.supxlabel("Error predicting wOBA (RMSE, lower is better). Dashed line: luck alone.",
                  fontsize=8.5, color=_MUTED)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=_SURFACE)
    plt.close(fig)


def run() -> dict:
    with db.connect() as conn:
        db.apply_schema(conn)
        history, league = load(conn)
        per_pa_sd = {int(s): float(v) for s, v in conn.execute(
            "SELECT season, stddev_samp(woba_value) FROM mlb.pa_clean "
            "WHERE counts_as_pa GROUP BY 1")}

        all_rows, fits, scores, comparisons = [], [], [], []
        for target in TARGETS:
            rows, result = holdout_rows(conn, history, league, target)
            all_rows.append(rows)
            fits.append({"target": target, "hitters_in_model": result["hitters"],
                         "seconds": result["seconds"], **result["diagnostics"]})
            scores.append(score(rows, per_pa_sd[target]).assign(target_season=str(target)))
            comparisons += [{"target_season": str(target), **compare(rows, b)} for b in BASELINES]
            d = result["diagnostics"]
            print(f"  {target}: fitted {result['hitters']} hitters in {result['seconds']:.0f}s, "
                  f"divergences {d['divergences']}, max r-hat {d['max_rhat']:.4f}, "
                  f"min effective draws {d['min_ess']:.0f}", flush=True)

        pooled = pd.concat(all_rows, ignore_index=True)
        scores.append(score(pooled, float(np.mean(list(per_pa_sd.values()))))
                      .assign(target_season="all"))
        comparisons += [{"target_season": "all", **compare(pooled, b)} for b in BASELINES]
        scores, comparisons = pd.concat(scores, ignore_index=True), pd.DataFrame(comparisons)

        conn.execute("TRUNCATE mlb.evaluation, mlb.evaluation_comparison")
        db.insert(conn, "mlb.evaluation", scores, ["target_season", "method", "hitters", "pa",
                                                   "rmse", "mae", "bias", "rank_corr",
                                                   "coverage_80", "luck_floor"])
        db.insert(conn, "mlb.evaluation_comparison", comparisons,
                  ["target_season", "versus", "rmse_gain", "gain_lo", "gain_hi", "share_better"])
        conn.commit()

    db.REPORTS.mkdir(exist_ok=True)
    (db.REPORTS / "evaluation.json").write_text(json.dumps(
        {"scores": scores.to_dict("records"), "comparisons": comparisons.to_dict("records"),
         "fits": fits}, indent=2), encoding="utf-8")
    draw(scores, db.REPORTS / "evaluation.png")

    for target, part in scores.groupby("target_season", sort=False):
        first = part.iloc[0]
        print(f"\n{target}: {first['hitters']} hitters, {first['pa']:,} PA   "
              f"(luck alone: RMSE {first['luck_floor']:.4f})")
        print(f"  {'method':<22}{'RMSE':>8}{'MAE':>8}{'bias':>9}{'rank':>7}{'80% cover':>11}")
        for row in part.sort_values("rmse").itertuples(index=False):
            cover = "" if row.coverage_80 is None or row.coverage_80 != row.coverage_80 \
                else f"{row.coverage_80:.0%}"
            rank = "" if row.rank_corr is None or row.rank_corr != row.rank_corr \
                else f"{row.rank_corr:.2f}"
            print(f"  {row.method:<22}{row.rmse:>8.4f}{row.mae:>8.4f}{row.bias:>+9.4f}"
                  f"{rank:>7}{cover:>11}")
        for row in comparisons[comparisons["target_season"] == target].itertuples(index=False):
            print(f"    model vs {row.versus:<20} gain {row.rmse_gain:+.4f}  "
                  f"(80%: {row.gain_lo:+.4f} to {row.gain_hi:+.4f})  "
                  f"better in {row.share_better:.0%} of resamples")
    unclean = [f for f in fits if f["divergences"] > 0 or f["max_rhat"] >= MAX_RHAT]
    print(f"\n{len(fits)} fits, {len(unclean)} with a divergence or r-hat >= {MAX_RHAT}")
    return {"scores": scores, "comparisons": comparisons, "fits": fits}
