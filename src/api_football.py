import os
from datetime import date
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

BASE_URL = "https://v3.football.api-sports.io"
API_KEY = os.getenv("API_FOOTBALL_KEY")


class APIFootballError(Exception):
    """Raised when API-Football returns an error or cannot be reached."""


def _headers() -> dict[str, str]:
    if not API_KEY:
        raise APIFootballError(
            "API_FOOTBALL_KEY is missing. Add it to the project .env file."
        )

    return {
        "x-apisports-key": API_KEY,
        "Accept": "application/json",
    }


def _get(endpoint: str, params: dict[str, Any] | None = None) -> list[dict]:
    """Send a GET request and return the API response list."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"

    try:
        response = requests.get(
            url,
            headers=_headers(),
            params=params,
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise APIFootballError(f"API request failed: {error}") from error

    payload = response.json()

    errors = payload.get("errors")
    if errors:
        raise APIFootballError(f"API-Football returned an error: {errors}")

    return payload.get("response", [])


def test_connection() -> bool:
    """Return True when the API key works."""
    _get("status")
    return True


def get_fixtures_by_date(
    fixture_date: date | str,
    league_id: int | None = None,
    season: int | None = None,
    timezone: str = "America/New_York",
) -> list[dict]:
    """Retrieve fixtures for a date, optionally filtered by league and season."""
    date_value = (
        fixture_date.isoformat()
        if isinstance(fixture_date, date)
        else str(fixture_date)
    )

    params: dict[str, Any] = {
        "date": date_value,
        "timezone": timezone,
    }

    if league_id is not None:
        params["league"] = league_id

    if season is not None:
        params["season"] = season

    return _get("fixtures", params)


def get_fixture(fixture_id: int) -> dict:
    """Retrieve full details for one fixture."""
    fixtures = _get("fixtures", {"id": fixture_id})
    return fixtures[0] if fixtures else {}


def get_lineups(fixture_id: int) -> list[dict]:
    """Retrieve confirmed lineups for one fixture."""
    return _get("fixtures/lineups", {"fixture": fixture_id})


def get_referee(fixture_id: int) -> str | None:
    """Retrieve the referee name from one fixture."""
    fixture = get_fixture(fixture_id)
    referee = fixture.get("fixture", {}).get("referee")

    if referee:
        return str(referee).strip()

    return None


def normalize_fixture(fixture: dict) -> dict:
    """Convert an API fixture into a simpler dashboard-friendly dictionary."""
    fixture_info = fixture.get("fixture", {})
    league = fixture.get("league", {})
    teams = fixture.get("teams", {})

    return {
        "fixture_id": fixture_info.get("id"),
        "date": fixture_info.get("date"),
        "status": fixture_info.get("status", {}).get("short"),
        "referee": fixture_info.get("referee"),
        "league_id": league.get("id"),
        "league": league.get("name"),
        "country": league.get("country"),
        "season": league.get("season"),
        "home_team_id": teams.get("home", {}).get("id"),
        "home_team": teams.get("home", {}).get("name"),
        "away_team_id": teams.get("away", {}).get("id"),
        "away_team": teams.get("away", {}).get("name"),
    }


def normalize_lineups(lineups: list[dict]) -> list[dict]:
    """Flatten API lineup data into one row per player."""
    rows = []

    for team_lineup in lineups:
        team = team_lineup.get("team", {})
        formation = team_lineup.get("formation")

        for lineup_type, api_key in (
            ("Starter", "startXI"),
            ("Substitute", "substitutes"),
        ):
            for item in team_lineup.get(api_key, []):
                player = item.get("player", {})

                rows.append(
                    {
                        "team_id": team.get("id"),
                        "team": team.get("name"),
                        "formation": formation,
                        "lineup_type": lineup_type,
                        "player_id": player.get("id"),
                        "player": player.get("name"),
                        "number": player.get("number"),
                        "position": player.get("pos"),
                        "grid": player.get("grid"),
                    }
                )

    return rows

def get_normalized_fixtures_by_date(
    fixture_date: date | str,
    league_id: int | None = None,
    season: int | None = None,
    timezone: str = "America/New_York",
) -> list[dict]:
    """Retrieve fixtures and convert them into dashboard-friendly dictionaries."""
    fixtures = get_fixtures_by_date(
        fixture_date=fixture_date,
        league_id=league_id,
        season=season,
        timezone=timezone,
    )

    return [normalize_fixture(fixture) for fixture in fixtures]


def get_fixture_label(fixture: dict) -> str:
    """Create a readable label for a fixture dropdown."""
    home_team = fixture.get("home_team") or "Unknown Home Team"
    away_team = fixture.get("away_team") or "Unknown Away Team"
    league = fixture.get("league") or "Unknown League"
    kickoff = fixture.get("date") or ""

    if kickoff:
        try:
            kickoff_time = kickoff[11:16]
        except (TypeError, IndexError):
            kickoff_time = ""
    else:
        kickoff_time = ""

    time_text = f" — {kickoff_time}" if kickoff_time else ""

    return f"{home_team} vs {away_team}{time_text} ({league})"


def get_fixture_lineup_data(fixture_id: int) -> dict:
    """Retrieve and normalize referee and lineup information for one fixture."""
    fixture = get_fixture(fixture_id)
    lineups = get_lineups(fixture_id)

    return {
        "fixture": normalize_fixture(fixture) if fixture else {},
        "referee": (
            fixture.get("fixture", {}).get("referee")
            if fixture
            else None
        ),
        "lineups": normalize_lineups(lineups),
        "lineups_available": bool(lineups),
    }