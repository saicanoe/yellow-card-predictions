import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from data_loader import data_path

API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
OUTPUT_FILE = "odds_api_latest.csv"
ODDS_COLUMNS = [
    "HomeTeam",
    "AwayTeam",
    "Line",
    "UnderOdds",
    "OverOdds",
    "LineupAdjustment",
]


def fetch_epl_odds():
    """Fetch EPL totals odds from The Odds API when ODDS_API_KEY is configured."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ODDS API: ODDS_API_KEY not set. Skipping API fetch.")
        return None

    params = urlencode(
        {
            "apiKey": api_key,
            "regions": "uk,eu,us",
            "markets": "totals",
            "oddsFormat": "decimal",
        }
    )
    url = f"{API_URL}?{params}"

    try:
        with urlopen(url, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        print(f"ODDS API: fetch failed with HTTP {error.code}.")
    except URLError as error:
        print(f"ODDS API: fetch failed: {error.reason}.")
    except TimeoutError:
        print("ODDS API: fetch timed out.")

    return None


def _extract_total_4_5_odds(event: dict) -> tuple[float | None, float | None]:
    """Return the best available under/over 4.5 prices from event bookmakers."""
    under_prices = []
    over_prices = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []):
                if float(outcome.get("point", 0)) != 4.5:
                    continue

                name = str(outcome.get("name", "")).strip().lower()
                price = outcome.get("price")
                if price is None:
                    continue
                if name == "under":
                    under_prices.append(float(price))
                elif name == "over":
                    over_prices.append(float(price))

    if not under_prices or not over_prices:
        return None, None

    return max(under_prices), max(over_prices)


def parse_odds_response(events) -> pd.DataFrame:
    """Parse The Odds API response into CardCast's odds input schema."""
    rows = []
    for event in events or []:
        under_odds, over_odds = _extract_total_4_5_odds(event)
        if under_odds is None or over_odds is None:
            continue

        rows.append(
            {
                "HomeTeam": event.get("home_team", ""),
                "AwayTeam": event.get("away_team", ""),
                "Line": 4.5,
                "UnderOdds": under_odds,
                "OverOdds": over_odds,
                "LineupAdjustment": 0.0,
            }
        )

    return pd.DataFrame(rows, columns=ODDS_COLUMNS)


def fetch_and_save_epl_odds() -> pd.DataFrame:
    """Fetch EPL odds and save the latest API snapshot if rows are available."""
    events = fetch_epl_odds()
    if events is None:
        return pd.DataFrame(columns=ODDS_COLUMNS)

    odds = parse_odds_response(events)
    if odds.empty:
        print("ODDS API: no EPL 4.5 totals odds returned.")
        return odds

    output_path = data_path(OUTPUT_FILE)
    odds.to_csv(output_path, index=False)
    print(f"ODDS API: saved {len(odds)} odds rows to data/{OUTPUT_FILE}.")
    return odds
