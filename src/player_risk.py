from pathlib import Path
import re
import unicodedata

import pandas as pd

from data_loader import normalize_team_name

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


def generate_live_player_card_risks(
    fixture: dict,
    lineup_rows: list[dict],
    live_prediction: dict,
) -> pd.DataFrame:
    """Score confirmed API starters against manually maintained profiles."""
    del fixture  # Reserved for later fixture-specific player features.

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

    matched = players["ProfileMatched"]
    players.loc[matched, "RiskScore"] = (
        _base_card_rate(players.loc[matched])
        + players.loc[matched, "Position"].apply(_position_boost)
        + players.loc[matched, "confidence"].apply(_match_confidence_boost)
        + players.loc[matched].apply(_card_total_boost, axis=1)
    ).round(3)
    players.loc[matched, "RiskTier"] = players.loc[matched, "RiskTier_profile"]

    players["RiskScore"] = pd.to_numeric(players["RiskScore"], errors="coerce")
    players = players.sort_values(
        ["ProfileMatched", "RiskScore", "Player"],
        ascending=[False, False, True],
        na_position="last",
    )
    return players[LIVE_OUTPUT_COLUMNS].reset_index(drop=True)


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
