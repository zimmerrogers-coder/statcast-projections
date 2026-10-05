"""Turns plate appearances into season totals and checks them.

Three steps: find out who is a pitcher (they are left out of the hitter
population), add up each hitter's season, and add up each season's league
numbers. Then the totals are checked against themselves and against Baseball
Savant's own published leaderboard.
"""

import csv
import io
import json
import urllib.request

import pandas as pd

from sample import db

PEOPLE = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"
LEADERBOARD = ("https://baseballsavant.mlb.com/leaderboard/expected_statistics"
               "?type=batter&year={season}&position=&team=&filterType=bip&min=1&csv=true")
FULL_SEASON_PA = (160_000, 195_000)
LEAGUE_WOBA = (0.295, 0.335)


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research sample)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8-sig")


def look_up_positions(conn, batch: int = 150) -> int:
    """Fill in primary position for every player that does not have one yet."""
    ids = [r[0] for r in conn.execute(
        "SELECT player_id FROM mlb.player WHERE primary_position IS NULL ORDER BY 1")]
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        people = json.loads(_get(PEOPLE.format(ids=",".join(map(str, chunk)))))["people"]
        for person in people:
            position = person.get("primaryPosition", {})
            conn.execute(
                "UPDATE mlb.player SET primary_position = %s, is_pitcher = %s WHERE player_id = %s",
                (position.get("abbreviation", "?"), position.get("code") == "1", person["id"]))
        conn.commit()
    # anyone the feed did not return is treated as a non-pitcher
    conn.execute("UPDATE mlb.player SET primary_position = '?', is_pitcher = false "
                 "WHERE primary_position IS NULL")
    conn.commit()
    return len(ids)


HITTER_SEASON = """
    INSERT INTO mlb.hitter_season
    SELECT pa.batter_id, pa.season,
           mode() WITHIN GROUP (ORDER BY pa.age_bat),
           mode() WITHIN GROUP (ORDER BY pa.bat_team),
           count(*) FILTER (WHERE event <> 'truncated_pa'),   -- inning ended mid-count: no PA
           count(*) FILTER (WHERE counts_as_pa),
           count(*) FILTER (WHERE counts_as_pa AND outcome = 'K'),
           count(*) FILTER (WHERE counts_as_pa AND outcome = 'BB'),
           count(*) FILTER (WHERE counts_as_pa AND outcome = 'HBP'),
           count(*) FILTER (WHERE outcome = 'IN_PLAY'),
           count(*) FILTER (WHERE outcome = 'IN_PLAY' AND xwoba_ball IS NOT NULL),
           coalesce(sum(xwoba_ball) FILTER (WHERE outcome = 'IN_PLAY'), 0),
           coalesce(sum(woba_value) FILTER (WHERE counts_as_pa), 0)
    FROM mlb.pa_clean pa
    JOIN mlb.player p ON p.player_id = pa.batter_id
    WHERE NOT p.is_pitcher
    GROUP BY pa.batter_id, pa.season
    HAVING count(*) FILTER (WHERE counts_as_pa) > 0
"""

LEAGUE_SEASON = """
    INSERT INTO mlb.league_season
    SELECT pa.season,
           count(DISTINCT pa.batter_id),
           count(*) FILTER (WHERE counts_as_pa),
           sum(woba_value) FILTER (WHERE counts_as_pa) / count(*) FILTER (WHERE counts_as_pa),
           count(*) FILTER (WHERE counts_as_pa AND outcome = 'K')::real
               / count(*) FILTER (WHERE counts_as_pa),
           count(*) FILTER (WHERE counts_as_pa AND outcome = 'BB')::real
               / count(*) FILTER (WHERE counts_as_pa),
           count(*) FILTER (WHERE counts_as_pa AND outcome = 'HBP')::real
               / count(*) FILTER (WHERE counts_as_pa),
           avg(xwoba_ball) FILTER (WHERE outcome = 'IN_PLAY'),
           avg(woba_value) FILTER (WHERE outcome = 'IN_PLAY' AND xwoba_ball IS NOT NULL),
           avg(woba_value) FILTER (WHERE counts_as_pa AND outcome = 'BB'),
           avg(woba_value) FILTER (WHERE counts_as_pa AND outcome = 'HBP'),
           count(*) FILTER (WHERE outcome = 'IN_PLAY' AND xwoba_ball IS NULL)::real
               / count(*) FILTER (WHERE outcome = 'IN_PLAY'),
           coalesce(avg(woba_value) FILTER (WHERE outcome = 'IN_PLAY' AND xwoba_ball IS NULL), 0),
           stddev_samp(woba_value) FILTER (WHERE outcome = 'IN_PLAY')
    FROM mlb.pa_clean pa
    JOIN mlb.player p ON p.player_id = pa.batter_id
    WHERE NOT p.is_pitcher
    GROUP BY pa.season
"""


def rebuild(conn) -> None:
    conn.execute("TRUNCATE mlb.hitter_season, mlb.league_season")
    conn.execute(HITTER_SEASON)
    conn.execute(LEAGUE_SEASON)
    conn.commit()


def internal_checks(conn) -> list[dict]:
    """Checks that need nothing but our own tables."""
    seasons = db.read(conn, "SELECT * FROM mlb.league_season ORDER BY season")
    broken = conn.execute(
        "SELECT count(*) FROM mlb.hitter_season WHERE pa <> k + bb + hbp + balls_in_play "
        "OR balls_measured > balls_in_play OR pa > all_pa").fetchone()[0]
    checks = [{"check": "PA = K + BB + HBP + in play, for every hitter-season",
               "detail": f"{broken} hitter-seasons break it", "ok": broken == 0}]
    for row in seasons.itertuples(index=False):
        checks.append({
            "check": f"{row.season}: plate appearances in the range of a full season",
            "detail": f"{row.pa:,}", "ok": FULL_SEASON_PA[0] <= row.pa <= FULL_SEASON_PA[1]})
        checks.append({
            "check": f"{row.season}: league wOBA plausible",
            "detail": f"{row.woba:.3f}", "ok": LEAGUE_WOBA[0] <= row.woba <= LEAGUE_WOBA[1]})
    gaps = conn.execute(
        "SELECT count(*) FROM (SELECT start_date, lag(end_date) OVER (PARTITION BY season "
        "ORDER BY start_date) AS previous_end FROM mlb.download_log) g "
        "WHERE previous_end IS NOT NULL AND start_date <> previous_end + 1").fetchone()[0]
    checks.append({"check": "no gaps between downloaded date ranges",
                   "detail": f"{gaps} gaps", "ok": gaps == 0})
    return checks


def leaderboard_check(conn, season: int, sample: int = 20, seed: int = 0) -> dict:
    """Compare a random 20 hitters with Baseball Savant's published season
    leaderboard: plate appearances, wOBA and expected wOBA. The leaderboard is
    built by MLB from the same pitches by their code, so agreement means our
    download and our counting are both right."""
    published = pd.DataFrame(list(csv.DictReader(io.StringIO(_get(LEADERBOARD.format(season=season))))))
    published = published.assign(
        batter_id=published["player_id"].astype(int),
        their_pa=pd.to_numeric(published["pa"], errors="coerce"),
        their_woba=pd.to_numeric(published["woba"], errors="coerce"),
        their_xwoba=pd.to_numeric(published["est_woba"], errors="coerce"))[
        ["batter_id", "their_pa", "their_woba", "their_xwoba"]].dropna()

    ours = db.read(conn, """
        SELECT h.batter_id, p.full_name, h.all_pa, h.pa, h.woba_num / h.pa AS woba,
               -- expected wOBA: results for strikeouts, walks and unmeasured balls,
               -- expected value for every measured ball
               (h.woba_num - m.measured_actual + h.xwobacon_sum) / h.pa AS xwoba
        FROM mlb.hitter_season h
        JOIN mlb.player p ON p.player_id = h.batter_id
        JOIN (SELECT batter_id, season,
                     coalesce(sum(woba_value) FILTER (WHERE outcome = 'IN_PLAY'
                              AND xwoba_ball IS NOT NULL), 0) AS measured_actual
              FROM mlb.pa_clean GROUP BY 1, 2) m USING (batter_id, season)
        WHERE h.season = %s AND h.pa >= 100""", (season,))
    joined = ours.merge(published, on="batter_id").sample(sample, random_state=seed)
    joined["pa_off"] = (joined["all_pa"] - joined["their_pa"]).abs()
    joined["woba_off"] = (joined["woba"] - joined["their_woba"]).abs()
    joined["xwoba_off"] = (joined["xwoba"] - joined["their_xwoba"]).abs()
    return {
        "season": season, "hitters": len(joined),
        "pa_exact": int((joined["pa_off"] == 0).sum()),
        "pa_within_2": int((joined["pa_off"] <= 2).sum()),
        "max_woba_off": float(joined["woba_off"].max()),
        "max_xwoba_off": float(joined["xwoba_off"].max()),
        # wOBA is allowed a wider gap than the others: Statcast's pitch data
        # carries rounded run values (0.7, 0.9, 1.25, 1.6, 2.0) while the
        # leaderboard uses each season's exact ones.
        "ok": bool((joined["pa_off"] <= 2).all() and joined["woba_off"].max() <= 0.010
                   and joined["xwoba_off"].max() <= 0.006),
        "table": joined,
    }


def run() -> bool:
    with db.connect() as conn:
        db.apply_schema(conn)
        looked_up = look_up_positions(conn)
        rebuild(conn)
        pitchers = conn.execute("SELECT count(*) FROM mlb.player WHERE is_pitcher").fetchone()[0]
        print(f"positions looked up for {looked_up} players; {pitchers} pitchers left out")
        print(db.read(conn, "SELECT season, hitters, pa, round(woba::numeric, 3) AS woba, "
                            "round(k_rate::numeric, 3) AS k, round(bb_rate::numeric, 3) AS bb, "
                            "round(xwobacon::numeric, 3) AS xwobacon, "
                            "round(unmeasured_share::numeric, 4) AS unmeasured "
                            "FROM mlb.league_season ORDER BY season").to_string(index=False))
        ok = True
        for check in internal_checks(conn):
            ok &= check["ok"]
            print(f"  {'ok  ' if check['ok'] else 'FAIL'} {check['check']}: {check['detail']}")
        for season in [r[0] for r in conn.execute("SELECT season FROM mlb.league_season ORDER BY 1")]:
            result = leaderboard_check(conn, season)
            ok &= result["ok"]
            print(f"  {'ok  ' if result['ok'] else 'FAIL'} {season} against Savant's leaderboard, "
                  f"{result['hitters']} hitters: PA exact for {result['pa_exact']}, within 2 for "
                  f"{result['pa_within_2']}; largest wOBA gap {result['max_woba_off']:.4f}, "
                  f"largest xwOBA gap {result['max_xwoba_off']:.4f}")
    return ok
