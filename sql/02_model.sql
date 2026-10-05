-- Model runs, projections and holdout scores. Safe to run more than once.

-- One row per model fit.
CREATE TABLE IF NOT EXISTS mlb.model_version (
    version        TEXT PRIMARY KEY,         -- e.g. 'holdout-2025', 'projection-2027'
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    target_season  SMALLINT NOT NULL,
    train_seasons  SMALLINT[] NOT NULL,
    hitters        INT NOT NULL,
    pa             INT NOT NULL,
    fit_seconds    REAL NOT NULL,
    diagnostics    JSONB NOT NULL            -- divergences, max r-hat, min effective sample size
);

-- League-level parameters of a fit, each with its 80% interval.
CREATE TABLE IF NOT EXISTS mlb.model_parameter (
    version    TEXT NOT NULL REFERENCES mlb.model_version ON DELETE CASCADE,
    parameter  TEXT NOT NULL,
    mean       REAL NOT NULL,
    lo_80      REAL NOT NULL,
    hi_80      REAL NOT NULL,
    PRIMARY KEY (version, parameter)
);

-- One row per hitter per fit: next season's projection.
CREATE TABLE IF NOT EXISTS mlb.projection (
    version      TEXT NOT NULL REFERENCES mlb.model_version ON DELETE CASCADE,
    batter_id    INT  NOT NULL,
    in_window    BOOLEAN NOT NULL,           -- false = no MLB history in the training seasons
    window_pa    INT,
    woba         REAL NOT NULL, woba_lo    REAL NOT NULL, woba_hi    REAL NOT NULL,  -- true talent
    season_lo    REAL NOT NULL, season_hi  REAL NOT NULL,  -- result over a 500-PA season
    k_rate       REAL NOT NULL, k_rate_lo  REAL NOT NULL, k_rate_hi  REAL NOT NULL,
    bb_rate      REAL NOT NULL, bb_rate_lo REAL NOT NULL, bb_rate_hi REAL NOT NULL,
    contact      REAL NOT NULL, contact_lo REAL NOT NULL, contact_hi REAL NOT NULL,
    marcel       REAL,                       -- Marcel's projection, for comparison
    PRIMARY KEY (version, batter_id)
);

-- Every method's score on a held-out season ('all' = the seasons pooled).
CREATE TABLE IF NOT EXISTS mlb.evaluation (
    target_season  TEXT NOT NULL,
    method         TEXT NOT NULL,
    hitters        INT  NOT NULL,
    pa             INT  NOT NULL,
    rmse           REAL NOT NULL,            -- weighted by plate appearances
    mae            REAL NOT NULL,
    bias           REAL NOT NULL,            -- predicted minus actual
    rank_corr      REAL,
    coverage_80    REAL,                     -- share of actual results inside the 80% range
    luck_floor     REAL NOT NULL,            -- error if true talent were known exactly
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (target_season, method)
);

-- The model against each other method.
CREATE TABLE IF NOT EXISTS mlb.evaluation_comparison (
    target_season  TEXT NOT NULL,
    versus         TEXT NOT NULL,
    rmse_gain      REAL NOT NULL,            -- other's RMSE minus the model's; positive = model better
    gain_lo        REAL NOT NULL,            -- 10th and 90th percentile when hitters are resampled
    gain_hi        REAL NOT NULL,
    share_better   REAL NOT NULL,
    PRIMARY KEY (target_season, versus)
);
