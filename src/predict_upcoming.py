import subprocess
import sys
import warnings
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        completed = subprocess.run([str(venv_python), *sys.argv], check=False)
        sys.exit(completed.returncode)
    raise

from bet_tracker import log_top_bets
from data_loader import (
    data_path,
    load_historical_matches,
    load_odds_input,
    load_referee_stats,
    load_upcoming_fixtures,
    load_upcoming_referees,
)
from features import add_upcoming_features, prepare_training_data
from model import add_predictions, train_models
from odds import add_betting_labels, add_odds_and_value, build_outputs
from odds_api import fetch_and_save_epl_odds
from player_risk import generate_player_card_risks, required_inputs_exist


def add_lineup_source_metadata(fixtures):
    """Flag fixtures that have lineup rows available for dashboard badges."""
    lineups_path = data_path("lineups.csv")
    fixtures = fixtures.copy()
    fixtures["LineupsLoaded"] = "NO"

    if not lineups_path.exists():
        return fixtures

    lineups = pd.read_csv(lineups_path)
    required_columns = {"Date", "HomeTeam", "AwayTeam"}
    if lineups.empty or not required_columns.issubset(lineups.columns):
        return fixtures

    def parse_lineup_dates(values):
        values = values.astype(str).str.strip()
        parsed = pd.to_datetime(values, dayfirst=True, errors="coerce")
        iso_mask = values.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
        parsed.loc[iso_mask] = pd.to_datetime(
            values.loc[iso_mask], format="%Y-%m-%d", errors="coerce"
        )
        return parsed.dt.strftime("%Y-%m-%d")

    lineup_keys = lineups[["Date", "HomeTeam", "AwayTeam"]].copy()
    lineup_keys["_match_date"] = parse_lineup_dates(lineup_keys["Date"])
    lineup_keys["HomeTeam"] = lineup_keys["HomeTeam"].astype(str).str.strip()
    lineup_keys["AwayTeam"] = lineup_keys["AwayTeam"].astype(str).str.strip()
    lineup_key_set = set(
        zip(lineup_keys["_match_date"], lineup_keys["HomeTeam"], lineup_keys["AwayTeam"])
    )

    fixture_dates = parse_lineup_dates(fixtures["Date"])
    fixture_keys = zip(
        fixture_dates,
        fixtures["HomeTeam"].astype(str).str.strip(),
        fixtures["AwayTeam"].astype(str).str.strip(),
    )
    fixtures["LineupsLoaded"] = [
        "YES" if fixture_key in lineup_key_set else "NO" for fixture_key in fixture_keys
    ]
    return fixtures


def print_run_summary(fixtures, output, top_bets, ultra_top_bets):
    """Print the same operational views the prototype used while running."""
    strong = fixtures[
        ((fixtures["predicted_cards"] > 5.2) & (fixtures["over_4_5_prob"] > 0.60))
        | ((fixtures["predicted_cards"] < 3.8) & (fixtures["over_4_5_prob"] < 0.40))
    ]

    print("\n STRONG SIGNALS:")
    print(strong[["HomeTeam", "AwayTeam", "predicted_cards", "edge", "signal"]])

    print("\n FULL OUTPUT:")
    print(output.to_string(index=False))

    print("\n TOP BETS:")
    print(
        top_bets[
            [
                "HomeTeam",
                "AwayTeam",
                "Referee",
                "predicted_cards",
                "over_4_5_prob",
                "edge",
                "signal",
                "confidence",
            ]
        ].to_string(index=False)
    )

    print("\n ULTRA VALUE:")
    print(
        ultra_top_bets[
            [
                "HomeTeam",
                "AwayTeam",
                "Referee",
                "predicted_cards",
                "total_cards_last5",
                "value_edge",
                "confidence",
                "ultra_value",
            ]
        ].to_string(index=False)
    )


def main():
    """Train from historical CSVs and score the upcoming fixture slate."""
    # The wide historical betting CSVs trigger pandas fragmentation warnings when
    # preserving the prototype's feature-building order. The warnings are noisy,
    # but the existing model behavior depends on this construction path.
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

    ref_stats = load_referee_stats()
    raw_matches = load_historical_matches()
    training_data, team_to_code, league_to_code, profiles = prepare_training_data(
        raw_matches, ref_stats
    )

    reg_model, clf_model = train_models(training_data)

    fixtures = load_upcoming_fixtures()
    fetch_and_save_epl_odds()
    upcoming_refs = load_upcoming_referees()
    if "Referee" in fixtures.columns:
        fixtures = fixtures.drop(columns=["Referee"])
    fixtures = fixtures.merge(upcoming_refs, on=["HomeTeam", "AwayTeam"], how="left")
    fixtures = add_upcoming_features(
        fixtures,
        training_data,
        ref_stats,
        team_to_code,
        league_to_code,
        profiles,
    )
    fixtures = add_predictions(fixtures, reg_model, clf_model)
    fixtures = add_odds_and_value(fixtures, load_odds_input(fixtures))
    fixtures = add_lineup_source_metadata(fixtures)
    fixtures = add_betting_labels(fixtures)

    output, top_bets, ultra_top_bets = build_outputs(fixtures)

    predictions_path = data_path("predictions.csv")
    top_bets_path = data_path("top_bets.csv")
    ultra_top_bets_path = data_path("ultra_top_bets.csv")
    output.to_csv(predictions_path, index=False)
    top_bets.to_csv(top_bets_path, index=False)
    ultra_top_bets.to_csv(ultra_top_bets_path, index=False)
    log_top_bets(top_bets)
    if required_inputs_exist():
        generate_player_card_risks()

    print_run_summary(fixtures, output, top_bets, ultra_top_bets)
    print("\nSaved:")
    print("data/predictions.csv")
    print("data/top_bets.csv")
    print("data/ultra_top_bets.csv")
    print("data/bet_tracking.csv")
    if required_inputs_exist():
        print("data/player_card_risks.csv")


if __name__ == "__main__":
    main()
