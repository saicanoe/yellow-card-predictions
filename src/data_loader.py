import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LEAGUES = ("E0", "SP1")
ODDS_COLUMNS = [
    "HomeTeam",
    "AwayTeam",
    "Line",
    "UnderOdds",
    "OverOdds",
    "LineupAdjustment",
    "OddsSource",
]
TEAM_ALIASES = {
    "brighton and hove albion": "brighton",
    "manchester city": "man city",
    "manchester united": "man united",
    "newcastle united": "newcastle",
    "nottingham forest": "nott'm forest",
    "tottenham hotspur": "tottenham",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves",
}


def data_path(filename: str) -> Path:
    """Return a path under the project data directory."""
    return DATA_DIR / filename


def load_historical_matches() -> pd.DataFrame:
    """Load only E0_ and SP1_ historical CSV files from data/."""
    files = [
        path
        for path in glob.glob(str(DATA_DIR / "*.csv"))
        if "E0_" in Path(path).name or "SP1_" in Path(path).name
    ]
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        # Kept for parity with the prototype's historical data assembly.
        frame["source_file"] = str(Path("data") / Path(path).name)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No E0_ or SP1_ historical CSV files found in data/.")

    return pd.concat(frames, ignore_index=True)


def load_referee_stats() -> pd.DataFrame:
    """Load referee profiles for England and Spain."""
    ref_epl = pd.read_csv(data_path("referees.csv"))
    ref_spain_path = data_path("referees_spain.csv")

    if ref_spain_path.exists():
        ref_spain = pd.read_csv(ref_spain_path)
        return pd.concat([ref_epl, ref_spain], ignore_index=True)

    return ref_epl


def load_upcoming_fixtures() -> pd.DataFrame:
    """Load upcoming fixtures and keep only supported leagues."""
    fixtures = pd.read_csv(data_path("upcoming_fixtures.csv"))
    fixtures = fixtures[fixtures["Div"].isin(LEAGUES)].copy()
    fixtures["HomeTeam"] = fixtures["HomeTeam"].str.strip()
    fixtures["AwayTeam"] = fixtures["AwayTeam"].str.strip()
    return fixtures


def load_upcoming_referees() -> pd.DataFrame:
    """Load manually entered referee assignments for upcoming fixtures."""
    upcoming_refs = pd.read_csv(
        data_path("upcoming_referees.csv"), encoding="utf-8-sig"
    )
    upcoming_refs.columns = upcoming_refs.columns.str.strip()
    upcoming_refs["HomeTeam"] = upcoming_refs["HomeTeam"].str.strip()
    upcoming_refs["AwayTeam"] = upcoming_refs["AwayTeam"].str.strip()
    upcoming_refs["Referee"] = upcoming_refs["Referee"].str.strip()
    return upcoming_refs


def normalize_team_name(team_name: str) -> str:
    """Normalize team names enough to match API names to fixture CSV names."""
    normalized = str(team_name).strip().lower()
    normalized = normalized.replace("&", "and")
    normalized = " ".join(normalized.replace(".", "").split())
    return TEAM_ALIASES.get(normalized, normalized)


def _empty_odds() -> pd.DataFrame:
    return pd.DataFrame(columns=ODDS_COLUMNS)


def _standardize_odds_frame(odds: pd.DataFrame) -> pd.DataFrame:
    """Keep the columns CardCast needs and coerce odds fields to numeric."""
    if odds.empty:
        return _empty_odds()

    odds = odds.copy()
    for column in ODDS_COLUMNS:
        if column not in odds.columns:
            if column == "LineupAdjustment":
                odds[column] = 0.0
            elif column == "OddsSource":
                odds[column] = pd.NA
            else:
                odds[column] = pd.NA

    odds = odds[ODDS_COLUMNS].copy()
    odds["HomeTeam"] = odds["HomeTeam"].astype(str).str.strip()
    odds["AwayTeam"] = odds["AwayTeam"].astype(str).str.strip()
    odds["OddsSource"] = odds["OddsSource"].astype(str).str.strip()
    for column in ["Line", "UnderOdds", "OverOdds", "LineupAdjustment"]:
        odds[column] = pd.to_numeric(odds[column], errors="coerce")
    return odds


def load_manual_odds_input() -> pd.DataFrame:
    """Load manually entered odds and lineup adjustments."""
    odds_path = data_path("odds_input.csv")
    if not odds_path.exists():
        return _empty_odds()
    odds = _standardize_odds_frame(pd.read_csv(odds_path))
    odds["OddsSource"] = "MANUAL ODDS"
    return odds


def load_api_odds_input() -> pd.DataFrame:
    """Load the latest saved API odds snapshot."""
    odds_path = data_path("odds_api_latest.csv")
    if not odds_path.exists():
        return _empty_odds()
    odds = _standardize_odds_frame(pd.read_csv(odds_path))
    odds["OddsSource"] = "API ODDS"
    return odds


def _match_odds_to_fixtures(fixtures: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Return odds rows rewritten with exact fixture team names after matching."""
    if fixtures.empty or odds.empty:
        return _empty_odds()

    fixture_keys = fixtures[["HomeTeam", "AwayTeam"]].drop_duplicates().copy()
    fixture_keys["home_key"] = fixture_keys["HomeTeam"].map(normalize_team_name)
    fixture_keys["away_key"] = fixture_keys["AwayTeam"].map(normalize_team_name)

    odds = odds.copy()
    odds["home_key"] = odds["HomeTeam"].map(normalize_team_name)
    odds["away_key"] = odds["AwayTeam"].map(normalize_team_name)

    matched = fixture_keys.merge(
        odds,
        on=["home_key", "away_key"],
        how="inner",
        suffixes=("_fixture", "_odds"),
    )
    if matched.empty:
        return _empty_odds()

    matched = matched.rename(
        columns={"HomeTeam_fixture": "HomeTeam", "AwayTeam_fixture": "AwayTeam"}
    )
    return matched[ODDS_COLUMNS].drop_duplicates(["HomeTeam", "AwayTeam"])


def load_odds_input(fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Load odds with priority: API snapshot, manual CSV, then model defaults.

    When fixtures are supplied, API and manual rows are matched by fixture and
    combined so manual CSV rows still cover matches missing from the API.
    """
    api_odds = load_api_odds_input()
    manual_odds = load_manual_odds_input()

    if fixtures is None:
        if not api_odds.empty:
            print(f"Odds source: API CSV used ({len(api_odds)} rows).")
            return api_odds
        if not manual_odds.empty:
            print(f"Odds source: manual CSV used ({len(manual_odds)} rows).")
            return manual_odds
        print("Odds source: defaults used (no API or manual CSV odds found).")
        return _empty_odds()

    api_matches = _match_odds_to_fixtures(fixtures, api_odds)
    manual_matches = _match_odds_to_fixtures(fixtures, manual_odds)

    api_keys = set(zip(api_matches["HomeTeam"], api_matches["AwayTeam"]))
    manual_fallback = manual_matches[
        ~manual_matches.apply(
            lambda row: (row["HomeTeam"], row["AwayTeam"]) in api_keys, axis=1
        )
    ].copy()

    combined = pd.concat([api_matches, manual_fallback], ignore_index=True)
    fixture_keys = set(zip(fixtures["HomeTeam"], fixtures["AwayTeam"]))
    combined_keys = set(zip(combined["HomeTeam"], combined["AwayTeam"]))
    default_count = len(fixture_keys - combined_keys)

    print(f"Odds source: API odds used for {len(api_matches)} matching fixtures.")
    print(f"Odds source: manual CSV odds used for {len(manual_fallback)} fixtures.")
    print(f"Odds source: default odds used for {default_count} fixtures.")
    return combined
