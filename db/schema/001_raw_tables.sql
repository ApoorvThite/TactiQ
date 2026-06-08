-- TactiQ Phase 1 — Raw Table Schema

-- Competitions reference table
CREATE TABLE IF NOT EXISTS competitions (
    competition_id      INTEGER,
    competition_name    TEXT NOT NULL,
    season_id           INTEGER,
    season_name         TEXT,
    country_name        TEXT,
    match_updated       TIMESTAMP,
    PRIMARY KEY (competition_id, season_id)
);

-- Teams reference table
CREATE TABLE IF NOT EXISTS teams (
    team_id     INTEGER PRIMARY KEY,
    team_name   TEXT NOT NULL,
    team_gender TEXT,
    country_name TEXT
);

-- Matches (one row per game)
CREATE TABLE IF NOT EXISTS matches (
    match_id            INTEGER PRIMARY KEY,
    competition_id      INTEGER,
    season_id           INTEGER,
    match_date          DATE,
    kick_off            TIME,
    home_team_id        INTEGER REFERENCES teams(team_id),
    away_team_id        INTEGER REFERENCES teams(team_id),
    home_score          INTEGER,
    away_score          INTEGER,
    match_status        TEXT,
    stadium_name        TEXT,
    referee_name        TEXT,
    FOREIGN KEY (competition_id, season_id) REFERENCES competitions(competition_id, season_id)
);

-- Match events (granular event log — core of the project)
CREATE TABLE IF NOT EXISTS match_events (
    event_id        UUID PRIMARY KEY,
    match_id        INTEGER REFERENCES matches(match_id),
    team_id         INTEGER REFERENCES teams(team_id),
    player_id       INTEGER,
    player_name     TEXT,
    event_index     INTEGER,
    event_type      TEXT NOT NULL,
    period          INTEGER,
    timestamp       INTERVAL,
    location_x      FLOAT,
    location_y      FLOAT,
    possession      INTEGER,
    possession_team_id INTEGER,
    play_pattern    TEXT,
    under_pressure  BOOLEAN,
    extra_data      JSONB
);

-- Kaggle historical international results
CREATE TABLE IF NOT EXISTS kaggle_results (
    id          SERIAL PRIMARY KEY,
    match_date  DATE,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    home_score  INTEGER,
    away_score  INTEGER,
    tournament  TEXT,
    city        TEXT,
    country     TEXT,
    neutral     BOOLEAN
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_events_match_id   ON match_events(match_id);
CREATE INDEX IF NOT EXISTS idx_events_team_id    ON match_events(team_id);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON match_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_extra      ON match_events USING GIN(extra_data);
CREATE INDEX IF NOT EXISTS idx_matches_date      ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_kaggle_teams      ON kaggle_results(home_team, away_team);
