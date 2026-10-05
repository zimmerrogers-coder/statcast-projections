"""Downloads public Statcast data and keeps one row per plate appearance.

Statcast's search returns every pitch. Only the pitch that ended a plate
appearance (the one with a value in `events`) carries the outcome, so that is
the only row kept. Each request covers a few days; every finished range is
logged, so a stopped download resumes without refetching.
"""

import csv
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

from sample import db

SEASONS = range(2021, 2027)
SEARCH = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R%7C"
          "&hfSea={season}%7C&player_type=batter&game_date_gt={start}&game_date_lt={end}"
          "&type=details")
CHUNK_DAYS = 4            # about 17,000 pitches; the search stops at 25,000 rows
ROW_LIMIT = 24_000        # a range this full is refetched in two halves
WORKERS = 3               # a few requests at a time, to be polite to a public site

STRIKEOUTS = {"strikeout", "strikeout_double_play"}

PA_COLUMNS = ["game_pk", "at_bat_number", "season", "game_date", "batter_id", "pitcher_id",
              "age_bat", "stand", "bat_team", "event", "outcome", "counts_as_pa", "woba_value",
              "launch_speed", "launch_angle", "xwoba_ball"]


def season_ranges(season: int, today: date | None = None) -> list[tuple[date, date]]:
    """Date ranges covering a regular season, generously (empty ranges are cheap)."""
    start, last = date(season, 3, 15), date(season, 10, 10)
    last = min(last, (today or date.today()) - timedelta(days=1))
    ranges = []
    while start <= last:
        end = min(start + timedelta(days=CHUNK_DAYS - 1), last)
        ranges.append((start, end))
        start = end + timedelta(days=1)
    return ranges


def fetch(season: int, start: date, end: date, attempts: int = 5) -> list[dict]:
    url = SEARCH.format(season=season, start=start.isoformat(), end=end.isoformat())
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research sample)"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                text = response.read().decode("utf-8-sig")
            return list(csv.DictReader(io.StringIO(text)))
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return []


def fetch_range(season: int, start: date, end: date) -> list[dict]:
    """Fetch a range, splitting it in two if it comes back at the row limit."""
    rows = fetch(season, start, end)
    if len(rows) >= ROW_LIMIT and start < end:
        middle = start + (end - start) // 2
        return (fetch_range(season, start, middle)
                + fetch_range(season, middle + timedelta(days=1), end))
    return rows


def _number(value):
    try:
        return float(value) if value not in ("", None) else None
    except ValueError:
        return None


def to_plate_appearances(rows: list[dict]) -> pd.DataFrame:
    """Keep the pitch that ended each plate appearance and name its outcome."""
    kept = []
    for r in rows:
        event = (r.get("events") or "").strip()
        if not event:
            continue
        counts = r.get("woba_denom") in ("1", "1.0")
        if event in STRIKEOUTS:
            outcome = "K"
        elif event == "walk":
            outcome = "BB"
        elif event == "hit_by_pitch":
            outcome = "HBP"
        elif counts:
            outcome = "IN_PLAY"
        else:
            outcome = "OTHER"
        top = (r.get("inning_topbot") or "").lower().startswith("top")
        last, _, first = (r.get("player_name") or "").partition(",")
        kept.append({
            "game_pk": int(r["game_pk"]), "at_bat_number": int(r["at_bat_number"]),
            "pitch_number": int(r.get("pitch_number") or 0),
            "season": int(r["game_year"]), "game_date": r["game_date"][:10],
            "batter_id": int(r["batter"]),
            "pitcher_id": int(r["pitcher"]) if r.get("pitcher") else None,
            "age_bat": int(float(r["age_bat"])) if r.get("age_bat") else None,
            "stand": (r.get("stand") or None),
            "bat_team": r.get("away_team") if top else r.get("home_team"),
            "event": event, "outcome": outcome, "counts_as_pa": counts,
            "woba_value": _number(r.get("woba_value")),
            "launch_speed": _number(r.get("launch_speed")),
            "launch_angle": _number(r.get("launch_angle")),
            "xwoba_ball": (_number(r.get("estimated_woba_using_speedangle"))
                           if outcome == "IN_PLAY" else None),
            "batter_name": f"{first.strip()} {last.strip()}".strip(),
        })
    frame = pd.DataFrame(kept)
    if frame.empty:
        return frame
    # a plate appearance has one ending; if two rows claim it, the later pitch wins
    return (frame.sort_values("pitch_number")
                 .drop_duplicates(["game_pk", "at_bat_number"], keep="last")
                 .drop(columns="pitch_number"))


def store(conn, season: int, start: date, end: date, rows: list[dict]) -> int:
    frame = to_plate_appearances(rows)
    if not frame.empty:
        names = frame.drop_duplicates("batter_id").rename(
            columns={"batter_id": "player_id", "batter_name": "full_name"})
        db.insert(conn, "mlb.player", names, ["player_id", "full_name"],
                  "ON CONFLICT (player_id) DO NOTHING")
        db.insert(conn, "mlb.plate_appearance", frame, PA_COLUMNS,
                  "ON CONFLICT (game_pk, at_bat_number) DO NOTHING")
    conn.execute(
        "INSERT INTO mlb.download_log (start_date, end_date, season, pitches_seen, pa_rows_kept) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
        (start, end, season, len(rows), len(frame)))
    conn.commit()
    return len(frame)


def run(seasons=SEASONS) -> None:
    with db.connect() as conn:
        db.apply_schema(conn)
        done = set(conn.execute("SELECT start_date, end_date FROM mlb.download_log").fetchall())
        todo = [(s, a, b) for s in seasons for a, b in season_ranges(s) if (a, b) not in done]
        print(f"{len(todo)} date ranges to fetch ({len(done)} already done)", flush=True)
        with ThreadPoolExecutor(WORKERS) as pool:
            for (season, start, end), rows in zip(
                    todo, pool.map(lambda job: fetch_range(*job), todo)):
                kept = store(conn, season, start, end, rows)
                print(f"  {start} to {end}: {len(rows):>6} pitches, {kept:>5} plate appearances",
                      flush=True)
        for season, pa in conn.execute(
                "SELECT season, count(*) FROM mlb.plate_appearance GROUP BY 1 ORDER BY 1"):
            print(f"{season}: {pa:,} plate-appearance rows")
