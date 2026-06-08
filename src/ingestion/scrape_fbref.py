"""Scrape FBref national team tactical stats for top international sides."""

import random
import time
from pathlib import Path

import pandas as pd
import requests

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "fbref_team_stats.csv"

# FBref squad IDs for national teams (from fbref.com squad pages)
TEAM_SQUAD_IDS = {
    "Argentina":     "f9fddd6e",
    "France":        "e8a5ae78",
    "Brazil":        "360cafbe",
    "England":       "cce30398",
    "Spain":         "fa37a51e",
    "Germany":       "adfa7c15",
    "Portugal":      "2f77b63c",
    "Netherlands":   "f5c7c3b2",
    "Italy":         "d7a486cd",
    "Belgium":       "fb08dac3",
    "Croatia":       "3a6e0f5f",
    "Uruguay":       "17d36c5e",
    "Morocco":       "7bf01ab3",
    "Senegal":       "0e9d38a7",
    "United States": "7f5df88c",
    "Mexico":        "b3bed136",
    "Japan":         "a3dc9bd0",
    "South Korea":   "ea2a3d3c",
    "Australia":     "2b1cc0ef",
    "Canada":        "fecb6498",
    "Ecuador":       "4a4e1a0d",
    "Denmark":       "a7c9c1e2",
    "Switzerland":   "b00a9a5e",
    "Poland":        "f41c11c8",
    "Serbia":        "f36e5908",
    "Cameroon":      "8c99b573",
    "Ghana":         "b33c6b2a",
    "Tunisia":       "9e1d9a5e",
    "Saudi Arabia":  "d024b1d3",
    "Iran":          "2929fa6e",
    "Qatar":         "b50e02e5",
    "Colombia":      "fb2a5a55",
    "Chile":         "a0a523c1",
    "Peru":          "3d3d7bc0",
    "Venezuela":     "b77da30b",
    "Wales":         "a4caa2e4",
    "Scotland":      "35bc109e",
    "Turkey":        "a8fd5ee4",
    "Austria":       "e1b9d43b",
    "Ukraine":       "1e57a4d6",
    "Hungary":       "38858a9f",
    "Czech Republic": "e5e7dd71",
    "Slovakia":      "d6ca6e12",
    "Slovenia":      "1b6c6ab6",
    "Albania":       "2c15bcc0",
    "Romania":       "adfe5b2a",
    "Georgia":       "e7cfbc45",
    "Jamaica":       "f37ca14a",
    "Panama":        "f7b42b0f",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

BASE_URL = "https://fbref.com/en/squads/{squad_id}/International-Results"
FBREF_HOME = "https://fbref.com/en/"


def _warm_up_session(session):
    """Hit the FBref homepage first so we get a valid session cookie."""
    try:
        session.get(FBREF_HOME, headers=HEADERS, timeout=20)
        time.sleep(random.uniform(3, 5))
    except Exception:
        pass


def _try_get_table(url, session, label):
    """Return the first useful DataFrame from an FBref page, or None on failure."""
    try:
        headers = {**HEADERS, "Referer": FBREF_HOME}
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(resp.text, flavor="lxml")
        if not tables:
            print(f"    [WARN] {label}: no tables found at {url}")
            return None
        return max(tables, key=lambda t: t.shape[0])
    except Exception as exc:
        print(f"    [FAIL] {label}: {exc}")
        return None


def scrape_team(team, squad_id, session):
    url = BASE_URL.format(squad_id=squad_id)
    df = _try_get_table(url, session, team)
    if df is None:
        return None
    df.columns = [
        "_".join(str(c) for c in col).strip() if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    df.insert(0, "team", team)
    df.insert(1, "squad_id", squad_id)
    return df


def main():
    session = requests.Session()
    print("  [INIT] Warming up session ...")
    _warm_up_session(session)

    results = []
    failed = []

    teams = list(TEAM_SQUAD_IDS.items())
    for i, (team, squad_id) in enumerate(teams, 1):
        print(f"  [{i:02d}/{len(teams)}] Scraping {team} ...", end=" ", flush=True)
        df = scrape_team(team, squad_id, session)
        if df is not None:
            results.append(df)
            print(f"OK ({len(df)} rows)")
        else:
            failed.append(team)
            print("FAILED")
        time.sleep(random.uniform(4, 8))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if results:
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv(OUTPUT_PATH, index=False)
    else:
        print("\n[ERROR] No data scraped — output file not written.")

    print("\n=== FBref Scrape Complete ===")
    print(f"Teams successfully scraped : {len(results)} / {len(teams)}")
    print(f"Teams failed               : {len(failed)}")
    if failed:
        print(f"Failed teams: {', '.join(failed)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
