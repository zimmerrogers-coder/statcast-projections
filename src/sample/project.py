"""The deliverable: 2027 projections for the hitters on Washington's 40-man roster.

Fits the model on 2024-2026, pulls the roster from MLB's public feed as it
stands when this runs, and writes a table and a chart.
"""

import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sample import db, evaluate  # noqa: E402
from sample.aggregate import _get  # noqa: E402

TARGET = 2027
TEAM_ID, TEAM_NAME = 120, "Washington Nationals"
ROSTER = "https://statsapi.mlb.com/api/v1/teams/{team}/roster?rosterType=40Man"
PEOPLE = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"

_INK, _SECONDARY, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
_BLUE, _LIGHT_BLUE, _NEUTRAL = "#2a78d6", "#9ec5f4", "#52514e"


def roster_hitters(team_id: int = TEAM_ID) -> pd.DataFrame:
    """Non-pitchers on the 40-man roster, with their age on July 1 of the target season."""
    roster = json.loads(_get(ROSTER.format(team=team_id)))["roster"]
    hitters = [r for r in roster if r["position"]["type"] != "Pitcher"]
    ids = ",".join(str(r["person"]["id"]) for r in hitters)
    born = {p["id"]: date.fromisoformat(p["birthDate"])
            for p in json.loads(_get(PEOPLE.format(ids=ids)))["people"]}
    midseason = date(TARGET, 7, 1)
    return pd.DataFrame([{
        "batter_id": r["person"]["id"], "name": r["person"]["fullName"],
        "position": r["position"]["abbreviation"],
        "age": (midseason - born[r["person"]["id"]]).days // 365,
    } for r in hitters]).set_index("batter_id")


def draw(table: pd.DataFrame, league_woba: float, path) -> None:
    ordered = table.sort_values("woba")
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(ordered) + 1.6), facecolor=_SURFACE)
    y = range(len(ordered))
    ax.axvline(league_woba, color=_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"2026 league average {league_woba:.3f}", (league_woba, len(ordered) - 0.35),
                xytext=(5, 0), textcoords="offset points", fontsize=8, color=_SECONDARY,
                va="center")
    ax.hlines(y, ordered["season_lo"], ordered["season_hi"], color=_LIGHT_BLUE, linewidth=2,
              zorder=2, label="80% range for a 500-PA season")
    ax.hlines(y, ordered["woba_lo"], ordered["woba_hi"], color=_BLUE, linewidth=5, zorder=3,
              label="80% range for true talent")
    ax.scatter(ordered["woba"], y, s=46, color=_SURFACE, edgecolor=_BLUE, linewidth=2, zorder=4,
               label="Projected wOBA")
    has_marcel = ordered["marcel"].notna()
    ax.scatter(ordered.loc[has_marcel, "marcel"], [i for i, m in zip(y, has_marcel) if m],
               marker="|", s=170, color=_NEUTRAL, linewidth=1.8, zorder=5, label="Marcel")
    labels = [f"{r.name}  ({int(r.window_pa):,} PA)" if r.window_pa > 0 else f"{r.name}  (no MLB PA)"
              for r in ordered.itertuples()]
    ax.set_yticks(list(y), labels, fontsize=9, color=_INK)
    ax.set_ylim(-0.7, len(ordered) + 0.1)
    ax.set_xlabel("Projected 2027 wOBA", fontsize=9, color=_MUTED)
    ax.set_title(f"{TEAM_NAME}: 2027 projections for 40-man hitters", loc="left", fontsize=11.5,
                 color=_INK, pad=12)
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=_MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_facecolor(_SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=_SURFACE)
    plt.close(fig)


def to_markdown(table: pd.DataFrame, fetched: date) -> str:
    lines = [
        f"# {TEAM_NAME}: 2027 hitter projections",
        "",
        f"40-man roster as of {fetched.isoformat()}, non-pitchers. Model fitted on 2024-2026 "
        "Statcast data.",
        "",
        "- **Projected wOBA** is the estimate of true talent; **talent range** is its 80% interval.",
        "- **500-PA range** is the 80% interval for the wOBA he would actually post over a "
        "500-PA season, which is wider because a season carries its own luck.",
        "- **MLB PA** is plate appearances in 2024-2026. A hitter with none gets the "
        "league-level projection for his age, and a wide range.",
        "",
        "| Hitter | Pos | Age | MLB PA | 2026 wOBA | Projected | Talent range | 500-PA range "
        "| K% | BB% | Contact | Marcel |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.sort_values("woba", ascending=False).itertuples():
        last = f"{r.woba_2026:.3f}" if r.woba_2026 == r.woba_2026 else "none"
        marcel = f"{r.marcel:.3f}" if r.marcel == r.marcel else "none"
        lines.append(
            f"| {r.name} | {r.position} | {r.age} | {int(r.window_pa):,} | {last} | "
            f"**{r.woba:.3f}** | {r.woba_lo:.3f} to {r.woba_hi:.3f} | "
            f"{r.season_lo:.3f} to {r.season_hi:.3f} | {r.k_rate:.1%} | {r.bb_rate:.1%} | "
            f"{r.contact:.3f} | {marcel} |")
    return "\n".join(lines) + "\n"


def run() -> pd.DataFrame:
    roster = roster_hitters()
    with db.connect() as conn:
        db.apply_schema(conn)
        history, league = evaluate.load(conn)
        window = history[history["season"].between(TARGET - 3, TARGET - 1)]
        unseen = roster[~roster.index.isin(window["batter_id"])]
        result = evaluate.fit_and_project(conn, history, league, TARGET, f"projection-{TARGET}",
                                          extra_hitters=unseen["age"].to_dict())
    d = result["diagnostics"]
    print(f"fitted {result['hitters']} hitters in {result['seconds']:.0f}s: divergences "
          f"{d['divergences']}, max r-hat {d['max_rhat']:.4f}, min effective draws {d['min_ess']:.0f}")

    last = history[history["season"] == TARGET - 1].set_index("batter_id")
    table = roster.join(result["projection"]).rename(columns={"Marcel": "marcel"})
    table["window_pa"] = table["window_pa"].fillna(0)
    table["woba_2026"] = (last["woba_num"] / last["pa"]).reindex(table.index)

    db.REPORTS.mkdir(exist_ok=True)
    (db.REPORTS / "nationals_2027.md").write_text(to_markdown(table, date.today()),
                                                  encoding="utf-8")
    draw(table, float(league.loc[TARGET - 1, "woba"]), db.REPORTS / "nationals_2027.png")
    shown = table.sort_values("woba", ascending=False)
    print(f"\n{'hitter':<24}{'age':>4}{'MLB PA':>8}{'2026':>7}{'proj':>7}   {'talent 80%':<16}"
          f"{'500-PA 80%':<16}{'Marcel':>7}")
    for r in shown.itertuples():
        last_woba = f"{r.woba_2026:.3f}" if r.woba_2026 == r.woba_2026 else "   -"
        marcel = f"{r.marcel:.3f}" if r.marcel == r.marcel else "    -"
        print(f"{r.name:<24}{r.age:>4}{int(r.window_pa):>8}{last_woba:>7}{r.woba:>7.3f}   "
              f"{r.woba_lo:.3f} to {r.woba_hi:.3f}  {r.season_lo:.3f} to {r.season_hi:.3f}  "
              f"{marcel:>7}")
    return table
