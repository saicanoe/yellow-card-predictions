from pathlib import Path
import re
import unicodedata

import pandas as pd

from data_loader import normalize_team_name
from player_stats_api import (
    PLAYER_STATS_CACHE_PATH,
    PlayerStatsAPIError,
    fetch_player_season_statistics,
    load_player_stats_cache,
    player_stats_cache_key,
    save_player_stats_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROFILES_PATH = DATA_DIR / "player_profiles.csv"
LINEUPS_PATH = DATA_DIR / "lineups.csv"
PREDICTIONS_PATH = DATA_DIR / "predictions.csv"
OUTPUT_PATH = DATA_DIR / "player_card_risks.csv"

OUTPUT_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "Player",
    "Team",
    "Position",
    "RiskScore",
    "RiskTier",
]

LIVE_OUTPUT_COLUMNS = [
    "Player",
    "Team",
    "Position",
    "RiskScore",
    "RiskTier",
    "ProfileMatched",
    "ProfileSource",
]

POSITION_BOOSTS = {
    "DM": 0.15,
    "CB": 0.10,
    "FB": 0.08,
    "CM": 0.05,
}


def required_inputs_exist() -> bool:
    """Return True when manual player profiles and lineups are available."""
    return PROFILES_PATH.exists() and LINEUPS_PATH.exists()


def load_player_profiles() -> pd.DataFrame:
    """Load manually maintained player card profiles."""
    return pd.read_csv(PROFILES_PATH)


def load_lineups() -> pd.DataFrame:
    """Load confirmed or expected lineups."""
    return pd.read_csv(LINEUPS_PATH)


def load_match_predictions() -> pd.DataFrame:
    """Load match-level predictions used to add fixture context."""
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH)


def _clean_text_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip()
    return frame


def _add_match_date_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "Date" not in frame.columns:
        frame["_match_date"] = pd.NaT
        return frame

    frame["_match_date"] = pd.to_datetime(
        frame["Date"], dayfirst=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return frame


def _starter_mask(lineups: pd.DataFrame) -> pd.Series:
    if "Starter" not in lineups.columns:
        return pd.Series(True, index=lineups.index)

    starter = lineups["Starter"].fillna("").astype(str).str.strip().str.upper()
    return starter.isin(["TRUE", "YES", "Y", "1", "STARTER"])


def _base_card_rate(players: pd.DataFrame) -> pd.Series:
    explicit_rate = pd.to_numeric(players.get("CardRatePer90"), errors="coerce")
    yellows = pd.to_numeric(players.get("YellowCards"), errors="coerce")
    minutes = pd.to_numeric(players.get("Minutes"), errors="coerce")
    calculated_rate = (yellows / minutes) * 90
    return explicit_rate.fillna(calculated_rate).fillna(0)


def _position_boost(position) -> float:
    return POSITION_BOOSTS.get(str(position).strip().upper(), 0.0)


def _match_confidence_boost(confidence) -> float:
    return 0.08 if str(confidence).strip().upper() == "HIGH CONFIDENCE" else 0.0


def _card_total_boost(row: pd.Series) -> float:
    card_total = pd.to_numeric(
        row.get("adjusted_cards", row.get("predicted_cards")), errors="coerce"
    )
    if pd.isna(card_total):
        return 0.0
    if card_total >= 5.2:
        return 0.15
    if card_total >= 4.5:
        return 0.08
    if card_total >= 3.8:
        return 0.03
    return 0.0


def _normalized_name(value) -> str:
    """Return an accent- and punctuation-insensitive lookup key."""
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    text = text.casefold().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _normalized_team(value) -> str:
    """Normalize known team aliases before creating a lookup key."""
    return _normalized_name(normalize_team_name(value))


def _select_api_profile(rows: list[dict], starter: pd.Series) -> dict | None:
    """Select a usable team statistics row for one lineup player."""
    if not rows:
        return None

    team_id = starter.get("team_id")
    team_key = _normalized_team(starter.get("Team"))
    candidates = [row for row in rows if isinstance(row, dict)]

    if pd.notna(team_id):
        team_matches = [
            row for row in candidates if str(row.get("team_id")) == str(team_id)
        ]
        if team_matches:
            candidates = team_matches
        elif team_key:
            candidates = [
                row
                for row in candidates
                if _normalized_team(row.get("team_name")) == team_key
            ]
        else:
            candidates = []
    elif team_key:
        team_matches = [
            row
            for row in candidates
            if _normalized_team(row.get("team_name")) == team_key
        ]
        if team_matches:
            candidates = team_matches

    for row in candidates:
        minutes = pd.to_numeric(row.get("minutes"), errors="coerce")
        card_rate = pd.to_numeric(row.get("card_rate_per_90"), errors="coerce")
        if pd.notna(minutes) and minutes > 0 and pd.notna(card_rate):
            return row

    return None


def _positive_int(value) -> int | None:
    """Coerce API identifiers while rejecting missing and non-positive values."""
    if isinstance(value, bool) or pd.isna(value):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def _risk_tier_from_score(score) -> str:
    """Assign a display tier to API-derived scores."""
    if pd.isna(score):
        return "UNMATCHED"
    if score >= 0.40:
        return "HIGH"
    if score >= 0.20:
        return "MEDIUM"
    return "LOW"


def generate_live_player_card_risks(
    fixture: dict,
    lineup_rows: list[dict],
    live_prediction: dict,
    fetch_missing: bool = False,
    max_api_fetches: int = 5,
    cache_path: Path = PLAYER_STATS_CACHE_PATH,
    profile_fetcher=None,
) -> pd.DataFrame:
    """Score starters from CSV first, then cache/API when explicitly requested."""

    if not lineup_rows:
        return pd.DataFrame(columns=LIVE_OUTPUT_COLUMNS)

    lineups = pd.DataFrame(lineup_rows)
    required_columns = {"player", "team", "position", "lineup_type"}
    if not required_columns.issubset(lineups.columns):
        return pd.DataFrame(columns=LIVE_OUTPUT_COLUMNS)

    starters = lineups[
        lineups["lineup_type"].fillna("").astype(str).str.strip().str.casefold()
        == "starter"
    ].copy()
    if starters.empty:
        return pd.DataFrame(columns=LIVE_OUTPUT_COLUMNS)

    starters = starters.rename(
        columns={"player": "Player", "team": "Team", "position": "Position_lineup"}
    )
    starters["_player_key"] = starters["Player"].map(_normalized_name)
    starters["_team_key"] = starters["Team"].map(_normalized_team)

    profiles = _clean_text_columns(
        load_player_profiles(), ["Player", "Team", "Position"]
    )
    profiles["_player_key"] = profiles["Player"].map(_normalized_name)
    profiles["_team_key"] = profiles["Team"].map(_normalized_team)
    profiles = profiles.drop_duplicates(["_player_key", "_team_key"], keep="first")
    profiles = profiles.rename(
        columns={
            "Player": "Player_profile",
            "Team": "Team_profile",
            "Position": "Position_profile",
            "RiskTier": "RiskTier_profile",
        }
    )

    players = starters.merge(
        profiles,
        on=["_player_key", "_team_key"],
        how="left",
        indicator=True,
    )
    players["ProfileMatched"] = players["_merge"].eq("both")
    players["ProfileSource"] = "Unmatched / not fetched"
    players.loc[players["ProfileMatched"], "ProfileSource"] = "CSV profile"
    players["Position"] = players["Position_profile"].fillna(
        players["Position_lineup"]
    )
    players["confidence"] = live_prediction.get("confidence")
    players["predicted_cards"] = live_prediction.get("predicted_cards")
    players["adjusted_cards"] = live_prediction.get(
        "adjusted_cards", live_prediction.get("predicted_cards")
    )
    players["RiskScore"] = pd.NA
    players["RiskTier"] = "UNMATCHED"

    cache = load_player_stats_cache(cache_path)
    cache_changed = False
    api_fetches = 0
    fetch_errors = []
    fetch_limit = max(0, int(max_api_fetches))
    fetcher = profile_fetcher or fetch_player_season_statistics
    season = fixture.get("season")
    league_id = fixture.get("league_id")

    for index in players.index[~players["ProfileMatched"]]:
        player_id = _positive_int(
            players.at[index, "player_id"] if "player_id" in players else None
        )
        season_id = _positive_int(season)
        league_identifier = _positive_int(league_id)
        if None in (player_id, season_id, league_identifier):
            continue

        cache_key = player_stats_cache_key(player_id, season_id, league_identifier)
        if cache_key in cache:
            api_rows = cache[cache_key]
            source = "API profile (cache)"
        elif fetch_missing and api_fetches < fetch_limit:
            api_fetches += 1
            try:
                api_rows = fetcher(player_id, season_id, league_identifier)
            except (PlayerStatsAPIError, ValueError) as error:
                fetch_errors.append(f"{players.at[index, 'Player']}: {error}")
                continue
            cache[cache_key] = api_rows
            cache_changed = True
            source = "API profile (live)"
        else:
            continue

        api_profile = _select_api_profile(api_rows, players.loc[index])
        if api_profile is None:
            continue

        players.at[index, "ProfileMatched"] = True
        players.at[index, "ProfileSource"] = source
        players.at[index, "CardRatePer90"] = api_profile["card_rate_per_90"]
        players.at[index, "YellowCards"] = api_profile["yellow_cards"]
        players.at[index, "Minutes"] = api_profile["minutes"]
        if api_profile.get("position"):
            players.at[index, "Position"] = api_profile["position"]

    if cache_changed:
        save_player_stats_cache(cache, cache_path)

    matched = players["ProfileMatched"]
    players.loc[matched, "RiskScore"] = (
        _base_card_rate(players.loc[matched])
        + players.loc[matched, "Position"].apply(_position_boost)
        + players.loc[matched, "confidence"].apply(_match_confidence_boost)
        + players.loc[matched].apply(_card_total_boost, axis=1)
    ).round(3)
    csv_matched = players["ProfileSource"].eq("CSV profile")
    players.loc[csv_matched, "RiskTier"] = players.loc[
        csv_matched, "RiskTier_profile"
    ]
    api_matched = matched & ~csv_matched
    players.loc[api_matched, "RiskTier"] = players.loc[
        api_matched, "RiskScore"
    ].apply(_risk_tier_from_score)

    players["RiskScore"] = pd.to_numeric(players["RiskScore"], errors="coerce")
    players = players.sort_values(
        ["ProfileMatched", "RiskScore", "Player"],
        ascending=[False, False, True],
        na_position="last",
    )
    output = players[LIVE_OUTPUT_COLUMNS].reset_index(drop=True)
    output.attrs["profile_counts"] = {
        "csv": int(output["ProfileSource"].eq("CSV profile").sum()),
        "cache": int(output["ProfileSource"].eq("API profile (cache)").sum()),
        "live_api": int(output["ProfileSource"].eq("API profile (live)").sum()),
        "unmatched": int((~output["ProfileMatched"]).sum()),
    }
    output.attrs["api_fetches"] = api_fetches
    output.attrs["fetch_errors"] = fetch_errors
    return output


def generate_player_card_risks() -> pd.DataFrame:
    """Create top-five player card-risk rankings for each available match."""
    if not required_inputs_exist():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    profiles = _clean_text_columns(
        load_player_profiles(), ["Player", "Team", "Position"]
    )
    lineups = _clean_text_columns(
        load_lineups(), ["HomeTeam", "AwayTeam", "Team", "Player", "Position"]
    )
    lineups = _add_match_date_key(lineups)

    if profiles.empty or lineups.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty.to_csv(OUTPUT_PATH, index=False)
        return empty

    lineups = lineups[_starter_mask(lineups)].copy()
    predictions = load_match_predictions()
    predictions = _clean_text_columns(predictions, ["HomeTeam", "AwayTeam"])
    predictions = _add_match_date_key(predictions)

    players = lineups.merge(
        profiles,
        on=["Player", "Team"],
        how="left",
        suffixes=("_lineup", "_profile"),
    )
    players["Position"] = players["Position_profile"].fillna(players["Position_lineup"])

    if not predictions.empty:
        match_context = predictions[
            [
                "_match_date",
                "HomeTeam",
                "AwayTeam",
                "predicted_cards",
                "adjusted_cards",
                "confidence",
            ]
        ].copy()
        players = players.merge(
            match_context,
            on=["_match_date", "HomeTeam", "AwayTeam"],
            how="left",
        )
    else:
        players["predicted_cards"] = pd.NA
        players["adjusted_cards"] = pd.NA
        players["confidence"] = pd.NA

    players["RiskTier"] = players["RiskTier"].fillna("UNKNOWN")
    players["RiskScore"] = (
        _base_card_rate(players)
        + players["Position"].apply(_position_boost)
        + players["confidence"].apply(_match_confidence_boost)
        + players.apply(_card_total_boost, axis=1)
    ).round(3)

    players = players.sort_values(
        ["Date", "HomeTeam", "AwayTeam", "RiskScore", "Player"],
        ascending=[True, True, True, False, True],
    )
    top_players = players.groupby(
        ["Date", "HomeTeam", "AwayTeam"], as_index=False
    ).head(5)
    output = top_players[OUTPUT_COLUMNS].copy()
    output.to_csv(OUTPUT_PATH, index=False)
    return output


if __name__ == "__main__":
    risks = generate_player_card_risks()
    print(risks.to_string(index=False))
    print("\nSaved:")
    print("data/player_card_risks.csv")
