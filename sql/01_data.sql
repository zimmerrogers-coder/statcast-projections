-- Data tables. Safe to run more than once.
-- Everything lives in the `mlb` schema of the advanced_metrics database.

CREATE SCHEMA IF NOT EXISTS mlb;

-- One row per date range already fetched from Statcast. A stopped download
-- picks up where it left off by skipping ranges that are in here.
CREATE TABLE IF NOT EXISTS mlb.download_log (
    start_date    DATE NOT NULL,
    end_date      DATE NOT NULL,
    season        SMALLINT NOT NULL,
    pitches_seen  INT NOT NULL,          -- rows Statcast returned
    pa_rows_kept  INT NOT NULL,          -- rows kept (the pitch that ended a plate appearance)
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (start_date, end_date)
);

CREATE TABLE IF NOT EXISTS mlb.player (
    player_id         INT PRIMARY KEY,   -- MLB's id
    full_name         TEXT NOT NULL,     -- "First Last"
    primary_position  TEXT,              -- from MLB's people feed; NULL until looked up
    is_pitcher        BOOLEAN
);

-- One row per plate appearance: the pitch on which it ended.
CREATE TABLE IF NOT EXISTS mlb.plate_appearance (
    game_pk        INT NOT NULL,
    at_bat_number  SMALLINT NOT NULL,
    season         SMALLINT NOT NULL,
    game_date      DATE NOT NULL,
    batter_id      INT NOT NULL,
    pitcher_id     INT,
    age_bat        SMALLINT,
    stand          CHAR(1),
    bat_team       TEXT,
    event          TEXT NOT NULL,        -- Statcast's `events`
    -- K, BB, HBP, IN_PLAY, or OTHER (intentional walk, sacrifice bunt, catcher's
    -- interference, an inning ending mid-count). OTHER is not a plate
    -- appearance for wOBA, which is what counts_as_pa records.
    outcome        TEXT NOT NULL CHECK (outcome IN ('K', 'BB', 'HBP', 'IN_PLAY', 'OTHER')),
    counts_as_pa   BOOLEAN NOT NULL,     -- Statcast's woba_denom = 1
    woba_value     REAL,
    launch_speed   REAL,
    launch_angle   REAL,
    xwoba_ball     REAL,                 -- Statcast's expected wOBA for a ball in play
    PRIMARY KEY (game_pk, at_bat_number)
);
CREATE INDEX IF NOT EXISTS pa_season_batter_idx ON mlb.plate_appearance (season, batter_id);

-- Plate appearances as the model counts them. Statcast's raw woba_value credits
-- a hitter who reached on an error or a fielder's choice as if he had singled
-- (0.9); standard wOBA gives those nothing. Catcher's interference is not a
-- plate appearance for wOBA. This view applies both corrections and leaves the
-- downloaded rows untouched.
CREATE OR REPLACE VIEW mlb.pa_clean AS
SELECT game_pk, at_bat_number, season, game_date, batter_id, pitcher_id, age_bat, stand,
       bat_team, event,
       CASE WHEN event = 'catcher_interf' THEN 'OTHER' ELSE outcome END AS outcome,
       (counts_as_pa AND event <> 'catcher_interf') AS counts_as_pa,
       CASE WHEN event IN ('field_error', 'fielders_choice') THEN 0 ELSE woba_value END
           AS woba_value,
       launch_speed, launch_angle,
       CASE WHEN event = 'catcher_interf' THEN NULL ELSE xwoba_ball END AS xwoba_ball
FROM mlb.plate_appearance;

-- One row per hitter per season: counts only. Rebuilt from plate_appearance.
CREATE TABLE IF NOT EXISTS mlb.hitter_season (
    batter_id       INT NOT NULL REFERENCES mlb.player,
    season          SMALLINT NOT NULL,
    age             SMALLINT,
    team            TEXT,                -- the team he batted for most
    all_pa          INT NOT NULL,        -- every trip, including the OTHER outcomes
    pa              INT NOT NULL,        -- trips that count for wOBA
    k               INT NOT NULL,
    bb              INT NOT NULL,
    hbp             INT NOT NULL,
    balls_in_play   INT NOT NULL,
    balls_measured  INT NOT NULL,        -- balls in play that have an expected wOBA
    xwobacon_sum    REAL NOT NULL,       -- sum of expected wOBA over measured balls
    woba_num        REAL NOT NULL,       -- sum of woba_value; wOBA = woba_num / pa
    PRIMARY KEY (batter_id, season)
);

-- League totals per season (non-pitchers only), and the numbers the model
-- needs that are the same for every hitter.
CREATE TABLE IF NOT EXISTS mlb.league_season (
    season            SMALLINT PRIMARY KEY,
    hitters           INT  NOT NULL,
    pa                INT  NOT NULL,
    woba              REAL NOT NULL,
    k_rate            REAL NOT NULL,
    bb_rate           REAL NOT NULL,
    hbp_rate          REAL NOT NULL,
    xwobacon          REAL NOT NULL,     -- expected wOBA per measured ball
    wobacon_measured  REAL NOT NULL,     -- what those same balls were actually worth
    walk_value        REAL NOT NULL,     -- that season's wOBA weight for a walk
    hbp_value         REAL NOT NULL,
    unmeasured_share  REAL NOT NULL,     -- share of balls in play with no expected wOBA
    unmeasured_value  REAL NOT NULL,     -- what those were actually worth
    in_play_sd        REAL NOT NULL      -- spread of woba_value across balls in play
);
