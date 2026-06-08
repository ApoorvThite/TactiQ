"""TactiQ Phase 2 — Full feature engineering pipeline orchestrator."""

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SQL_FILES = [
    ROOT / "db/schema/002_feature_tables.sql",
    ROOT / "src/features/compute_match_features.sql",
    ROOT / "src/features/compute_rolling_form.sql",
    ROOT / "src/features/compute_style_profiles.sql",
]

LABELS = [
    "Create feature tables",
    "Compute match-team features",
    "Compute rolling form",
    "Compute team style profiles",
]


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "tactiq"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def run_sql_file(conn, path, label):
    sql = path.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print(f"  [OK] {label}")


def fetch_one(cur, sql, params=None):
    cur.execute(sql, params or [])
    return cur.fetchone()[0]


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or [])
    return cur.fetchall()


def print_summary(conn):
    cur = conn.cursor()
    sep = "=" * 60

    print(f"\n{sep}")
    print(" TACTIQ — Phase 2 Feature Pipeline Complete")
    print(sep)

    # match_team_features stats
    total      = fetch_one(cur, "SELECT COUNT(*) FROM match_team_features")
    ppda_rows  = fetch_one(cur, "SELECT COUNT(*) FROM match_team_features WHERE ppda != 999")
    xg_rows    = fetch_one(cur, "SELECT COUNT(*) FROM match_team_features WHERE xg_created > 0")
    avg_poss   = fetch_one(cur, "SELECT ROUND(AVG(possession_pct)::NUMERIC,2) FROM match_team_features")
    avg_xg     = fetch_one(cur, "SELECT ROUND(AVG(xg_created)::NUMERIC,3) FROM match_team_features")

    print("\nMATCH_TEAM_FEATURES")
    print(f"  Total rows               : {total}   (expected ~460 = 230 matches × 2 teams)")
    print(f"  Rows with ppda computed  : {ppda_rows}")
    print(f"  Rows with xg_created > 0 : {xg_rows}")
    print(f"  Avg possession_pct       : {avg_poss}%  (should be near 50%)")
    print(f"  Avg xg_created per match : {avg_xg}")

    # team_style_profiles stats
    n_teams    = fetch_one(cur, "SELECT COUNT(*) FROM team_style_profiles")
    avg_matches= fetch_one(cur, "SELECT ROUND(AVG(matches_played)::NUMERIC,1) FROM team_style_profiles")
    ppda_min   = fetch_one(cur, "SELECT ROUND(MIN(avg_ppda)::NUMERIC,3) FROM team_style_profiles")
    ppda_max   = fetch_one(cur, "SELECT ROUND(MAX(avg_ppda)::NUMERIC,3) FROM team_style_profiles")
    xgr_min    = fetch_one(cur, "SELECT ROUND(MIN(avg_xg_ratio)::NUMERIC,3) FROM team_style_profiles")
    xgr_max    = fetch_one(cur, "SELECT ROUND(MAX(avg_xg_ratio)::NUMERIC,3) FROM team_style_profiles")
    poss_min   = fetch_one(cur, "SELECT ROUND(MIN(avg_possession_pct)::NUMERIC,2) FROM team_style_profiles")
    poss_max   = fetch_one(cur, "SELECT ROUND(MAX(avg_possession_pct)::NUMERIC,2) FROM team_style_profiles")

    print("\nTEAM_STYLE_PROFILES")
    print(f"  Teams profiled           : {n_teams}")
    print(f"  Avg matches per team     : {avg_matches}")
    print(f"  PPDA range               : {ppda_min} (min) — {ppda_max} (max)")
    print(f"  xG ratio range           : {xgr_min} (min) — {xgr_max} (max)")
    print(f"  Possession range         : {poss_min}% — {poss_max}%")

    # Top 5 by PPDA (lowest = most aggressive press)
    print("\nTOP 5 TEAMS BY PPDA (most aggressive pressing — lower = more pressing)")
    rows = fetch_all(cur, """
        SELECT team_name, avg_ppda FROM team_style_profiles
        WHERE avg_ppda IS NOT NULL
        ORDER BY avg_ppda ASC LIMIT 5
    """)
    for i, (name, val) in enumerate(rows, 1):
        print(f"  {i}. {name:<25} : {val}")

    # Top 5 by xG ratio
    print("\nTOP 5 TEAMS BY XG_RATIO (best attack/defense balance)")
    rows = fetch_all(cur, """
        SELECT team_name, avg_xg_ratio FROM team_style_profiles
        WHERE avg_xg_ratio IS NOT NULL
        ORDER BY avg_xg_ratio DESC LIMIT 5
    """)
    for i, (name, val) in enumerate(rows, 1):
        print(f"  {i}. {name:<25} : {val}")

    # Top 5 by progressive carry pct
    print("\nTOP 5 TEAMS BY PROGRESSIVE CARRY PCT")
    rows = fetch_all(cur, """
        SELECT team_name, avg_progressive_carry_pct FROM team_style_profiles
        WHERE avg_progressive_carry_pct IS NOT NULL
        ORDER BY avg_progressive_carry_pct DESC LIMIT 5
    """)
    for i, (name, val) in enumerate(rows, 1):
        print(f"  {i}. {name:<25} : {val}%")

    # NULL audit
    roll_nulls = fetch_one(cur, "SELECT COUNT(*) FROM match_team_features WHERE rolling_xg_created_5 IS NULL")
    ppda_nulls = fetch_one(cur, "SELECT COUNT(*) FROM match_team_features WHERE ppda IS NULL")
    xgr_nulls  = fetch_one(cur, "SELECT COUNT(*) FROM match_team_features WHERE xg_ratio IS NULL")

    print("\nNULL AUDIT")
    print(f"  rolling_xg_created_5 nulls : {roll_nulls}  (first 5 matches per team expected)")
    print(f"  ppda nulls                 : {ppda_nulls}  (should be 0)")
    print(f"  xg_ratio nulls             : {xgr_nulls}  (0-xg-conceded games expected)")

    print(f"\n{sep}\n")
    cur.close()


def main():
    conn = get_connection()
    print("\nTactiQ Phase 2 — Feature Pipeline")
    print("-" * 40)

    for path, label in zip(SQL_FILES, LABELS):
        print(f"  Running: {label} ...")
        run_sql_file(conn, path, label)

    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
