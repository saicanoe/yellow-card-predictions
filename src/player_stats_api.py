"""API-Football player season-statistics retrieval and normalization."""

from typing import Any

from api_football import APIFootballError, _get


class PlayerStatsAPIError(APIFootballError):
    """Raised when player statistics cannot be retrieved or normalized."""


PLAYER_STAT_COLUMNS = [
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "appearances",
    "starts",
    "minutes",
    "yellow_cards",
    "second_yellow_cards",
    "red_cards",
    "fouls_committed",
    "card_rate_per_90",
]


def _safe_int(value: Any) -> int:
    """Convert an API statistic to an integer, defaulting missing values to zero."""
    if value is None or isinstance(value, bool):
        return 0

    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _normalize_statistic(player: dict, statistic: dict) -> dict:
    """Normalize one API-Football statistics object into a flat row."""
    team = statistic.get("team") or {}
    games = statistic.get("games") or {}
    cards = statistic.get("cards") or {}
    fouls = statistic.get("fouls") or {}

    minutes = _safe_int(games.get("minutes"))
    yellow_cards = _safe_int(cards.get("yellow"))
    card_rate_per_90 = (
        round((yellow_cards / minutes) * 90, 3)
        if minutes > 0
        else 0.0
    )

    return {
        "player_id": player.get("id"),
        "player_name": player.get("name"),
        "team_id": team.get("id"),
        "team_name": team.get("name"),
        "position": games.get("position"),
        "appearances": _safe_int(games.get("appearences")),
        "starts": _safe_int(games.get("lineups")),
        "minutes": minutes,
        "yellow_cards": yellow_cards,
        "second_yellow_cards": _safe_int(cards.get("yellowred")),
        "red_cards": _safe_int(cards.get("red")),
        "fouls_committed": _safe_int(fouls.get("committed")),
        "card_rate_per_90": card_rate_per_90,
    }


def normalize_player_statistics(response: list[dict]) -> list[dict]:
    """Normalize API player responses into one row per team statistics entry."""
    normalized = []

    for item in response or []:
        if not isinstance(item, dict):
            continue
        player = item.get("player") or {}
        statistics = item.get("statistics") or []

        if not isinstance(player, dict) or not isinstance(statistics, list):
            continue

        for statistic in statistics:
            if not isinstance(statistic, dict):
                continue
            normalized.append(_normalize_statistic(player, statistic))

    return normalized


def fetch_player_season_statistics(
    player_id: int,
    season: int,
    league_id: int,
) -> list[dict]:
    """Fetch and normalize one player's season statistics for a league."""
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (player_id, season, league_id)
    ):
        raise ValueError("player_id, season, and league_id must be positive integers.")

    try:
        response = _get(
            "players",
            {
                "id": player_id,
                "season": season,
                "league": league_id,
            },
        )
    except APIFootballError as error:
        raise PlayerStatsAPIError(
            f"Could not fetch season statistics for player {player_id}: {error}"
        ) from error

    try:
        return normalize_player_statistics(response)
    except (AttributeError, TypeError) as error:
        raise PlayerStatsAPIError(
            f"Could not normalize season statistics for player {player_id}."
        ) from error


def _run_smoke_test() -> None:
    """Exercise normalization without requiring an API key or network request."""
    sample_response = [
        {
            "player": {"id": 123, "name": "Test Player"},
            "statistics": [
                {
                    "team": {"id": 42, "name": "Arsenal"},
                    "games": {
                        "appearences": 10,
                        "lineups": 8,
                        "minutes": 900,
                        "position": "Midfielder",
                    },
                    "cards": {"yellow": 3, "yellowred": 1, "red": None},
                    "fouls": {"committed": 14},
                },
                {
                    "team": {"id": 99, "name": "Partial FC"},
                    "games": {"minutes": 0},
                },
            ],
        }
    ]

    rows = normalize_player_statistics(sample_response)
    assert len(rows) == 2
    assert list(rows[0]) == PLAYER_STAT_COLUMNS
    assert rows[0]["card_rate_per_90"] == 0.3
    assert rows[1]["card_rate_per_90"] == 0.0
    assert rows[1]["yellow_cards"] == 0
    print("player_stats_api smoke test passed")


if __name__ == "__main__":
    _run_smoke_test()
