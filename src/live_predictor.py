import re
import unicodedata

import pandas as pd

from data_loader import (
    load_historical_matches,
    load_referee_stats,
    normalize_team_name,
)
from features import add_upcoming_features, prepare_training_data
from model import add_predictions, train_models


LEAGUE_ID_TO_DIV = {
    39: "E0",    # Premier League
    140: "SP1",  # La Liga
}


class LivePredictionError(Exception):
    """Raised when a selected API fixture cannot be scored."""


def normalize_lookup_key(value: str | None) -> str:
    """Create a loose comparison key for API and historical names."""
    if not value:
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.casefold().replace("&", "and")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return " ".join(text.split())


def strip_referee_country(referee: str | None) -> str | None:
    """Remove country text returned after an API referee name."""
    if not referee:
        return None

    return str(referee).split(",", maxsplit=1)[0].strip() or None


def resolve_division(fixture: dict) -> str:
    """Map an API-Football league to the model's division code."""
    league_id = fixture.get("league_id")

    if league_id in LEAGUE_ID_TO_DIV:
        return LEAGUE_ID_TO_DIV[league_id]

    raise LivePredictionError(
        "Live prediction currently supports only the Premier League "
        "and La Liga."
    )


def get_team_candidates(
    raw_matches: pd.DataFrame,
    division: str,
) -> list[str]:
    """Return historical team names for one division."""
    league_matches = raw_matches[
        raw_matches["Div"] == division
    ]

    teams = set(league_matches["HomeTeam"].dropna())
    teams.update(league_matches["AwayTeam"].dropna())

    return sorted(teams)


def resolve_team_name(
    api_team_name: str,
    candidates: list[str],
) -> str:
    """Match an API team name to its historical CSV spelling."""
    normalized_api_name = normalize_team_name(api_team_name)
    api_key = normalize_lookup_key(normalized_api_name)

    exact_matches = [
        candidate
        for candidate in candidates
        if normalize_lookup_key(
            normalize_team_name(candidate)
        ) == api_key
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    partial_matches = [
        candidate
        for candidate in candidates
        if api_key
        and (
            api_key
            in normalize_lookup_key(normalize_team_name(candidate))
            or normalize_lookup_key(normalize_team_name(candidate))
            in api_key
        )
    ]

    if len(partial_matches) == 1:
        return partial_matches[0]

    raise LivePredictionError(
        f'Could not match API team "{api_team_name}" '
        "to the historical dataset."
    )


def resolve_referee_name(
    api_referee: str | None,
    referee_stats: pd.DataFrame,
) -> tuple[str | None, bool]:
    """Match an API referee to an existing referee profile."""
    cleaned_referee = strip_referee_country(api_referee)

    if not cleaned_referee:
        return None, False

    referee_names = (
        referee_stats["Referee"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    api_key = normalize_lookup_key(cleaned_referee)

    matches = [
        referee_name
        for referee_name in referee_names
        if normalize_lookup_key(referee_name) == api_key
    ]

    if len(matches) == 1:
        return matches[0], True

    return cleaned_referee, False


def generate_live_prediction(fixture: dict) -> dict:
    """
    Train the existing models and score one normalized API fixture.

    The fixture should come from api_football.normalize_fixture().
    """
    home_team_api = str(
        fixture.get("home_team") or ""
    ).strip()

    away_team_api = str(
        fixture.get("away_team") or ""
    ).strip()

    if not home_team_api or not away_team_api:
        raise LivePredictionError(
            "The selected fixture is missing team information."
        )

    division = resolve_division(fixture)

    raw_matches = load_historical_matches()
    referee_stats = load_referee_stats()

    team_candidates = get_team_candidates(
        raw_matches,
        division,
    )

    home_team_model = resolve_team_name(
        home_team_api,
        team_candidates,
    )

    away_team_model = resolve_team_name(
        away_team_api,
        team_candidates,
    )

    referee_api = fixture.get("referee")

    referee_model, referee_profile_found = resolve_referee_name(
        referee_api,
        referee_stats,
    )

    (
        training_data,
        team_to_code,
        league_to_code,
        profiles,
    ) = prepare_training_data(
        raw_matches.copy(),
        referee_stats.copy(),
    )

    regression_model, classification_model = train_models(
        training_data
    )

    fixture_frame = pd.DataFrame(
        [
            {
                "Date": fixture.get("date"),
                "HomeTeam": home_team_model,
                "AwayTeam": away_team_model,
                "Div": division,
                "Referee": referee_model,
            }
        ]
    )

    featured_fixture = add_upcoming_features(
        fixture_frame,
        training_data,
        referee_stats,
        team_to_code,
        league_to_code,
        profiles,
    )

    if featured_fixture.empty:
        raise LivePredictionError(
            "The fixture could not be scored because its "
            "historical feature data is incomplete."
        )

    predicted_fixture = add_predictions(
        featured_fixture,
        regression_model,
        classification_model,
    ).iloc[0]

    over_probability = float(
        predicted_fixture["over_4_5_prob"]
    )

    return {
        "fixture_id": fixture.get("fixture_id"),
        "home_team_api": home_team_api,
        "away_team_api": away_team_api,
        "home_team_model": home_team_model,
        "away_team_model": away_team_model,
        "league": fixture.get("league"),
        "division": division,
        "referee_api": strip_referee_country(referee_api),
        "referee_model": referee_model,
        "referee_profile_found": referee_profile_found,
        "predicted_cards": float(
            predicted_fixture["predicted_cards"]
        ),
        "over_4_5_probability": over_probability,
        "under_4_5_probability": 1.0 - over_probability,
    }