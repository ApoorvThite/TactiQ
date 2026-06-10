"""Phase 1 data validation — prints a full integrity report to stdout."""

import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

CRITICAL_MIN_EVENTS = 100_000  # lower bound for a partial load; full load > 500k


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "tactiq"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def q(cur, sql, params=None):
    cur.execute(sql, params)
    return cur.fetchall()


def q1(cur, sql, params=None):
    rows = q(cur, sql, params)
    return rows[0][0] if rows else None


def main():
    warnings = []

    try:
        conn = get_connection()
    except psycopg2.Error as exc:
        print(f"FATAL: Cannot connect to database — {exc}")
        sys.exit(1)

    cur = conn.cursor()
    SEP = "=" * 60

    print(SEP)
    print(" TACTIQ — Phase 1 Data Validation Report")
    print(SEP)

    # --- Row counts ---
    tables = ["competitions", "teams", "matches", "match_events", "kaggle_results"]
    counts = {}
    print("\nTABLE ROW COUNTS")
    for t in tables:
        n = q1(cur, f"SELECT COUNT(*) FROM {t}")
        counts[t] = n
        print(f"  {t:<20}: {n:,} rows")

    # --- Top event types ---
    print("\nMATCH EVENTS — TOP 10 EVENT TYPES")
    rows = q(cur, """
        SELECT event_type, COUNT(*) AS n
        FROM match_events
        GROUP BY event_type
        ORDER BY n DESC
        LIMIT 10
    """)
    for etype, n in rows:
        print(f"  {etype:<25}: {n:,}")

    # --- Competitions breakdown ---
    print("\nCOMPETITIONS LOADED")
    rows = q(cur, """
        SELECT c.competition_name, c.season_name,
               COUNT(DISTINCT m.match_id)  AS matches,
               COUNT(e.event_id)           AS events
        FROM competitions c
        LEFT JOIN matches m
            ON m.competition_id = c.competition_id
           AND m.season_id      = c.season_id
        LEFT JOIN match_events e ON e.match_id = m.match_id
        GROUP BY c.competition_name, c.season_name
        ORDER BY c.competition_name, c.season_name
    """)
    for comp, season, m, e in rows:
        label = f"{comp} {season}"
        print(f"  {label:<35}: {m} matches, {e:,} events")

    # --- Kaggle summary ---
    print("\nKAGGLE RESULTS")
    kmin = q1(cur, "SELECT MIN(match_date) FROM kaggle_results")
    kmax = q1(cur, "SELECT MAX(match_date) FROM kaggle_results")
    khome = q1(cur, "SELECT COUNT(DISTINCT home_team) FROM kaggle_results")
    kaway = q1(cur, "SELECT COUNT(DISTINCT away_team) FROM kaggle_results")
    ktour = q1(cur, "SELECT COUNT(DISTINCT tournament) FROM kaggle_results")
    print(f"  Date range        : {kmin} to {kmax}")
    print(f"  Unique home teams : {khome}")
    print(f"  Unique away teams : {kaway}")
    print(f"  Unique tournaments: {ktour}")

    # --- Null checks ---
    print("\nNULL CHECK (key columns)")
    null_checks = [
        ("match_events", "event_type"),
        ("match_events", "team_id"),
        ("matches", "match_date"),
        ("matches", "home_team_id"),
    ]
    for tbl, col in null_checks:
        n = q1(cur, f"SELECT COUNT(*) FROM {tbl} WHERE {col} IS NULL")
        print(f"  {tbl}.{col:<25} nulls: {n}")

    # --- Data quality ---
    print("\nDATA QUALITY")
    total_matches = counts["matches"] or 1
    total_events = counts["match_events"] or 1

    scored = q1(cur, "SELECT COUNT(*) FROM matches WHERE home_score IS NOT NULL AND away_score IS NOT NULL")
    with_loc = q1(cur, "SELECT COUNT(*) FROM match_events WHERE location_x IS NOT NULL")
    with_pres = q1(cur, "SELECT COUNT(*) FROM match_events WHERE under_pressure IS NOT NULL")

    def pct(a, b):
        return f"{100*a//b}%" if b else "N/A"
    print(f"  Matches with both scores present : {scored} / {total_matches} ({pct(scored, total_matches)})")
    print(f"  Events with location data        : {with_loc:,} / {total_events:,} ({pct(with_loc, total_events)})")
    print(f"  Events with under_pressure flag  : {with_pres:,} / {total_events:,} ({pct(with_pres, total_events)})")

    # --- Critical checks ---
    if counts["match_events"] == 0:
        warnings.append("CRITICAL: match_events is empty. Re-run: python src/ingestion/load_statsbomb.py")
    elif counts["match_events"] < CRITICAL_MIN_EVENTS:
        warnings.append(
            f"WARNING: match_events has only {counts['match_events']:,} rows "
            f"(expected > {CRITICAL_MIN_EVENTS:,}). Load may be incomplete."
        )
    if counts["kaggle_results"] == 0:
        warnings.append("CRITICAL: kaggle_results is empty. Place results.csv in data/raw/ and re-run load_kaggle_results.py")
    if counts["competitions"] == 0:
        warnings.append("CRITICAL: competitions is empty. Re-run: python src/ingestion/load_statsbomb.py")

    print()
    print(SEP)
    if warnings:
        for w in warnings:
            print(f" ⚠  {w}")
    else:
        print(" All checks passed. Phase 1 data ready for feature engineering.")
    print(SEP)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
