"""Load StatsBomb open data into PostgreSQL (competitions, teams, matches, events)."""

import json
import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from statsbombpy import sb
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# (competition_id, season_id) pairs to ingest from StatsBomb free data
TARGET_COMPETITIONS = {
    # FIFA World Cup
    (43, 3):  "FIFA World Cup 2018",
    (43, 106): "FIFA World Cup 2022",
    # UEFA Euro
    (55, 43): "UEFA Euro 2020",
    (55, 282): "UEFA Euro 2024",
    # Copa América
    (223, 44): "Copa América 2021",
    (223, 281): "Copa América 2024",
    # AFC Asian Cup 2023 — may not be in free tier; handled gracefully
    (36, 76): "AFC Asian Cup 2023",
}

def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# Top-level event fields that have dedicated columns — excluded from extra_data
_SKIP_KEYS = {
    "id", "match_id", "team", "player", "index", "type",
    "period", "timestamp", "location", "possession",
    "possession_team", "play_pattern", "under_pressure",
}


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "tactiq"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def _safe_id(obj):
    """Extract id from a dict-like object or return None."""
    if isinstance(obj, dict):
        return obj.get("id")
    return None


def _safe_name(obj):
    if isinstance(obj, dict):
        return obj.get("name")
    if isinstance(obj, str) and obj:
        return obj
    return None


def load_competitions(cur, comp_df):
    rows = comp_df[["competition_id", "competition_name", "season_id",
                     "season_name", "country_name"]].drop_duplicates().itertuples(index=False)
    cur.executemany(
        """
        INSERT INTO competitions (competition_id, competition_name, season_id, season_name, country_name)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        [(r.competition_id, r.competition_name, r.season_id, r.season_name, r.country_name)
         for r in rows],
    )


def load_teams(cur, team_rows):
    cur.executemany(
        """
        INSERT INTO teams (team_id, team_name)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """,
        team_rows,
    )


def load_match(cur, m):
    home = m.get("home_team", {})
    away = m.get("away_team", {})
    stadium = m.get("stadium", {})
    referee = m.get("referee", {})
    cur.execute(
        """
        INSERT INTO matches
            (match_id, competition_id, season_id, match_date, kick_off,
             home_team_id, away_team_id, home_score, away_score,
             match_status, stadium_name, referee_name)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
        """,
        (
            m["match_id"],
            m["competition_id"],
            m["season_id"],
            m.get("match_date"),
            m.get("kick_off"),
            _safe_id(home),
            _safe_id(away),
            m.get("home_score"),
            m.get("away_score"),
            m.get("match_status"),
            _safe_name(stadium),
            _safe_name(referee),
        ),
    )


def load_events(cur, events_df, match_id):
    rows = []
    skipped = 0
    for _, e in events_df.iterrows():
        event_type = _safe_name(e.get("type"))
        if not event_type:
            skipped += 1
            continue

        loc = e.get("location")
        loc_x = loc[0] if isinstance(loc, list) and len(loc) >= 2 else None
        loc_y = loc[1] if isinstance(loc, list) and len(loc) >= 2 else None

        team = e.get("team", {})
        player = e.get("player", {})
        poss_team = e.get("possession_team", {})

        extra = _sanitize({k: v for k, v in e.items() if k not in _SKIP_KEYS and v is not None})

        rows.append((
            str(e["id"]),
            match_id,
            _safe_id(team),
            _safe_id(player),
            _safe_name(player),
            e.get("index"),
            event_type,
            e.get("period"),
            str(e["timestamp"]) if e.get("timestamp") is not None else None,
            loc_x,
            loc_y,
            e.get("possession"),
            _safe_id(poss_team),
            _safe_name(e.get("play_pattern")),
            bool(e["under_pressure"]) if e.get("under_pressure") is not None else None,
            json.dumps(extra, default=str),
        ))

    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO match_events
            (event_id, match_id, team_id, player_id, player_name,
             event_index, event_type, period, timestamp,
             location_x, location_y, possession, possession_team_id,
             play_pattern, under_pressure, extra_data)
        VALUES %s
        ON CONFLICT DO NOTHING
        """,
        rows,
        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)",
        page_size=100,
    )
    return len(rows), skipped


def main():
    all_comps = sb.competitions()

    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    total_competitions = 0
    total_matches = 0
    total_teams = set()
    total_events = 0
    all_dates = []

    for (comp_id, season_id), label in TARGET_COMPETITIONS.items():
        subset = all_comps[
            (all_comps["competition_id"] == comp_id) &
            (all_comps["season_id"] == season_id)
        ]
        if subset.empty:
            print(f"  [SKIP] {label} — not found in StatsBomb free data")
            continue

        print(f"\n  Loading: {label}")
        load_competitions(cur, subset)
        total_competitions += 1

        matches_df = sb.matches(competition_id=comp_id, season_id=season_id)

        team_rows = set()
        for _, m in matches_df.iterrows():
            for side in ["home_team", "away_team"]:
                t = m.get(side, {})
                if isinstance(t, dict) and t.get("home_team_id" if side == "home_team" else "away_team_id"):
                    tid = t.get("home_team_id") or t.get("away_team_id")
                    tname = t.get("home_team_name") or t.get("away_team_name") or ""
                    team_rows.add((tid, tname))

        # statsbombpy returns flat columns for home/away team
        for _, m in matches_df.iterrows():
            for prefix in ["home", "away"]:
                tid = m.get(f"{prefix}_team_id")
                tname = m.get(f"{prefix}_team_name", "")
                if tid is not None:
                    team_rows.add((int(tid), str(tname)))
                    total_teams.add(int(tid))

        load_teams(cur, list(team_rows))

        match_records = matches_df.to_dict(orient="records")
        for m in match_records:
            # Rebuild minimal dicts load_match expects
            m["home_team"] = {"id": m.get("home_team_id"), "name": m.get("home_team_name")}
            m["away_team"] = {"id": m.get("away_team_id"), "name": m.get("away_team_name")}
            m["stadium"] = {"name": m.get("stadium_name")}
            m["referee"] = {"name": m.get("referee")}
            m["competition_id"] = comp_id
            m["season_id"] = season_id
            load_match(cur, m)
            if m.get("match_date"):
                all_dates.append(str(m["match_date"]))

        total_matches += len(match_records)
        conn.commit()

        total_skipped = 0
        match_ids = matches_df["match_id"].tolist()
        for mid in tqdm(match_ids, desc=f"    Events [{label}]"):
            for attempt in range(3):
                try:
                    # Fresh connection per match — isolates crashes
                    mconn = get_connection()
                    mconn.autocommit = False
                    mcur = mconn.cursor()
                    ev = sb.events(match_id=mid)
                    n, skipped = load_events(mcur, ev, mid)
                    mconn.commit()
                    mcur.close()
                    mconn.close()
                    total_events += n
                    total_skipped += skipped
                    break
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                    print(f"    [RETRY {attempt+1}/3] match {mid}: {exc}")
                    try:
                        mconn.close()
                    except Exception:
                        pass
                    if attempt == 2:
                        print(f"    [SKIP] match {mid} failed after 3 attempts")
                except Exception as exc:
                    print(f"    [WARN] match {mid} events failed: {exc}")
                    try:
                        mconn.close()
                    except Exception:
                        pass
                    break
        if total_skipped:
            print(f"    (skipped {total_skipped} typeless meta-events)")

    cur.close()
    conn.close()

    date_range = f"{min(all_dates)} to {max(all_dates)}" if all_dates else "N/A"
    print("\n=== StatsBomb Load Complete ===")
    print(f"Competitions loaded : {total_competitions}")
    print(f"Matches loaded      : {total_matches}")
    print(f"Teams loaded        : {len(total_teams)}")
    print(f"Events loaded       : {total_events}")
    print(f"Date range          : {date_range}")


if __name__ == "__main__":
    main()
