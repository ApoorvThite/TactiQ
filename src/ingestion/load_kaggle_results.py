"""Load Kaggle international football results CSV into PostgreSQL."""

import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "results.csv"


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "tactiq"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: results.csv not found at {CSV_PATH}", file=sys.stderr)
        print("Download from Kaggle: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].dt.year >= 2010].copy()
    df = df.dropna(subset=["home_team", "away_team"])

    rows = [
        (
            row["date"].date() if pd.notna(row["date"]) else None,
            row.get("home_team"),
            row.get("away_team"),
            int(row["home_score"]) if pd.notna(row.get("home_score")) else None,
            int(row["away_score"]) if pd.notna(row.get("away_score")) else None,
            row.get("tournament"),
            row.get("city"),
            row.get("country"),
            bool(row["neutral"]) if pd.notna(row.get("neutral")) else None,
        )
        for _, row in df.iterrows()
    ]

    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO kaggle_results
            (match_date, home_team, away_team, home_score, away_score,
             tournament, city, country, neutral)
        VALUES %s
        ON CONFLICT DO NOTHING
        """,
        rows,
        page_size=500,
    )
    conn.commit()
    cur.close()
    conn.close()

    dates = df["date"].dropna()
    unique_teams = pd.concat([df["home_team"], df["away_team"]]).nunique()
    unique_tournaments = df["tournament"].nunique()

    print("=== Kaggle Results Load Complete ===")
    print(f"Rows loaded    : {len(rows)}")
    print(f"Date range     : {dates.min().date()} to {dates.max().date()}")
    print(f"Unique teams   : {unique_teams}")
    print(f"Tournaments    : {unique_tournaments} unique")


if __name__ == "__main__":
    main()
