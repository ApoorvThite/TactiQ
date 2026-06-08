"""Phase 2 Step 1 — Explore raw match_events structure before feature engineering."""

import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "tactiq"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def run_table(cur, label, sql, params=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("=" * 60)
    cur.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    col_widths = [max(len(c), max((len(str(r[i])) for r in rows), default=0)) for i, c in enumerate(cols)]
    header = " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols))
    print("  " + header)
    print("  " + "-" * len(header))
    for row in rows:
        print("  " + " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))


def run_json(cur, label, sql, params=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("=" * 60)
    cur.execute(sql, params or [])
    row = cur.fetchone()
    if row and row[0]:
        val = row[0]
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                pass
        print(json.dumps(val, indent=2, default=str))
    else:
        print("  (no result)")


def main():
    conn = get_connection()
    cur = conn.cursor()

    # 1a. Event type distribution
    run_table(cur, "1a. Event types and counts",
        "SELECT event_type, COUNT(*) AS n FROM match_events GROUP BY event_type ORDER BY n DESC")

    # 1b-1e: extra_data samples per event type (using RealDictCursor approach via json cast)
    for event_type, label in [
        ("Pass",     "1b. extra_data sample — Pass"),
        ("Shot",     "1c. extra_data sample — Shot"),
        ("Pressure", "1d. extra_data sample — Pressure"),
        ("Carry",    "1e. extra_data sample — Carry"),
    ]:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print("=" * 60)
        cur.execute(
            "SELECT extra_data::text FROM match_events WHERE event_type = %s "
            "AND extra_data IS NOT NULL AND extra_data != '{}'::jsonb LIMIT 1",
            (event_type,)
        )
        row = cur.fetchone()
        if row and row[0]:
            try:
                print(json.dumps(json.loads(row[0]), indent=2, default=str))
            except Exception:
                print(row[0])
        else:
            print("  (no rows found)")

    # 1f. Unique teams
    run_table(cur, "1f. Unique team count in match_events",
        "SELECT COUNT(DISTINCT team_id) AS unique_teams FROM match_events")

    # 1g. Sample matches with team names
    run_table(cur, "1g. Sample matches with team names",
        """
        SELECT m.match_id, m.match_date,
               t1.team_name AS home_team, m.home_score,
               t2.team_name AS away_team, m.away_score,
               c.competition_name, c.season_name
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.team_id
        JOIN teams t2 ON m.away_team_id = t2.team_id
        JOIN competitions c USING (competition_id, season_id)
        ORDER BY m.match_date
        LIMIT 10
        """)

    # Extra: verify flat JSONB paths
    print(f"\n{'='*60}")
    print("  EXTRA: Verified flat JSONB path checks")
    print("=" * 60)
    path_checks = [
        ("Shot xG (flat)",         "Shot",     "extra_data->>'shot_statsbomb_xg'"),
        ("Pass outcome (flat)",    "Pass",     "extra_data->>'pass_outcome'"),
        ("Pass end_location",      "Pass",     "extra_data->'pass_end_location'"),
        ("Pass length",            "Pass",     "extra_data->>'pass_length'"),
        ("Carry end_location",     "Carry",    "extra_data->'carry_end_location'"),
        ("Shot type (flat)",       "Shot",     "extra_data->>'shot_type'"),
        ("Shot outcome (flat)",    "Shot",     "extra_data->>'shot_outcome'"),
    ]
    for path_label, event_type, path in path_checks:
        cur.execute(
            f"SELECT {path} AS val FROM match_events WHERE event_type = %s "
            "AND extra_data IS NOT NULL LIMIT 3",
            (event_type,)
        )
        rows = cur.fetchall()
        print(f"\n  {path_label} [{path}]")
        for r in rows:
            print(f"    {r[0]}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
