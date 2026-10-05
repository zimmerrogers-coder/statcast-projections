"""Parsing Statcast rows into plate appearances, and the date ranges requested."""

from datetime import date

from sample import download


def pitch(at_bat, number, events="", denom="", **extra):
    row = {"game_pk": "1", "at_bat_number": str(at_bat), "pitch_number": str(number),
           "game_year": "2025", "game_date": "2025-06-03", "batter": "100", "pitcher": "200",
           "age_bat": "27", "stand": "R", "inning_topbot": "Top", "home_team": "WSH",
           "away_team": "NYM", "player_name": "Wood, James", "events": events,
           "woba_denom": denom, "woba_value": "", "launch_speed": "", "launch_angle": "",
           "estimated_woba_using_speedangle": ""}
    row.update(extra)
    return row


def test_only_the_ending_pitch_is_kept_and_named():
    rows = [
        pitch(1, 1), pitch(1, 2), pitch(1, 3, "strikeout", "1", woba_value="0"),
        pitch(2, 1, "walk", "1", woba_value="0.7"),
        pitch(3, 1, "hit_by_pitch", "1", woba_value="0.7"),
        pitch(4, 1, "single", "1", woba_value="0.9", launch_speed="101.2", launch_angle="12",
              estimated_woba_using_speedangle="0.61"),
        pitch(5, 1, "intent_walk", "0", woba_value="0.4"),
        pitch(6, 1, "sac_bunt", "0"),
        pitch(7, 1, "strikeout_double_play", "1", woba_value="0"),
    ]
    frame = download.to_plate_appearances(rows).sort_values("at_bat_number")
    assert frame["outcome"].tolist() == ["K", "BB", "HBP", "IN_PLAY", "OTHER", "OTHER", "K"]
    assert frame["counts_as_pa"].tolist() == [True, True, True, True, False, False, True]
    single = frame[frame["at_bat_number"] == 4].iloc[0]
    assert single["xwoba_ball"] == 0.61 and single["launch_speed"] == 101.2
    assert frame[frame["outcome"] != "IN_PLAY"]["xwoba_ball"].isna().all()
    # Statcast writes "Last, First"; top of the inning means the away team is batting
    assert single["batter_name"] == "James Wood" and single["bat_team"] == "NYM"
    assert single["age_bat"] == 27 and single["season"] == 2025


def test_one_row_per_plate_appearance():
    rows = [pitch(1, 2, "caught_stealing_2b", ""), pitch(1, 5, "single", "1", woba_value="0.9")]
    frame = download.to_plate_appearances(rows)
    assert len(frame) == 1 and frame["event"].iloc[0] == "single"


def test_nothing_to_keep():
    assert download.to_plate_appearances([pitch(1, 1), pitch(1, 2)]).empty


def test_season_ranges_cover_the_season_without_gaps_or_overlap():
    ranges = download.season_ranges(2024, today=date(2026, 1, 1))
    assert ranges[0][0] == date(2024, 3, 15) and ranges[-1][1] == date(2024, 10, 10)
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert (start - end).days == 1
    assert all((end - start).days < download.CHUNK_DAYS for start, end in ranges)


def test_a_season_in_progress_stops_at_yesterday():
    ranges = download.season_ranges(2026, today=date(2026, 5, 1))
    assert ranges[-1][1] == date(2026, 4, 30)
