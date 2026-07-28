import pandas as pd

OUTPUT_COLUMNS = [
    "Date",
    "Div",
    "HomeTeam",
    "AwayTeam",
    "Referee",
    "avg_total_cards",
    "over_4_5_rate",
    "predicted_cards",
    "adjusted_cards",
    "over_4_5_prob",
    "edge",
    "signal",
    "confidence",
    "under_model_prob",
    "under_book_prob",
    "value_edge",
    "value_bet",
    "ultra_value",
    "OddsSource",
    "LineupsLoaded",
    "UnderOdds",
]


def add_odds_and_value(fixtures: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Merge odds input and calculate adjusted cards plus value edge."""
    fixtures = fixtures.merge(odds, on=["HomeTeam", "AwayTeam"], how="left")

    if "OddsSource" not in fixtures.columns:
        fixtures["OddsSource"] = pd.NA

    fixtures["Line"] = fixtures["Line"].fillna(4.5)
    fixtures["UnderOdds"] = fixtures["UnderOdds"].fillna(1.80)
    fixtures["OverOdds"] = fixtures["OverOdds"].fillna(2.00)
    fixtures["LineupAdjustment"] = fixtures["LineupAdjustment"].fillna(0.00)
    fixtures["OddsSource"] = fixtures["OddsSource"].fillna("DEFAULT ODDS")

    fixtures["adjusted_cards"] = (
        fixtures["predicted_cards"] + fixtures["LineupAdjustment"]
    )
    fixtures["under_book_prob"] = 1 / fixtures["UnderOdds"]
    fixtures["over_book_prob"] = 1 / fixtures["OverOdds"]
    fixtures["under_model_prob"] = 1 - fixtures["over_4_5_prob"]
    fixtures["value_edge"] = fixtures["under_model_prob"] - fixtures["under_book_prob"]
    fixtures["value_bet"] = fixtures["value_edge"].apply(
        lambda edge: "YES" if edge > 0.05 else "NO"
    )

    fixtures["line"] = 4.5
    fixtures["edge"] = fixtures["adjusted_cards"] - fixtures["Line"]
    return fixtures


def get_signal(row: pd.Series) -> str:
    """Convert model card total and probability into the prototype signal labels."""
    if row["adjusted_cards"] > 5.2 and row["over_4_5_prob"] > 0.60:
        return "STRONG OVER"

    if row["adjusted_cards"] < 3.8 and row["over_4_5_prob"] < 0.40:
        return "STRONG UNDER"

    if (row["adjusted_cards"] > 4.5 and row["over_4_5_prob"] < 0.50) or (
        row["adjusted_cards"] < 4.5 and row["over_4_5_prob"] > 0.50
    ):
        return "NO BET"

    if row["adjusted_cards"] > 4.5:
        return "LEAN OVER"

    return "LEAN UNDER"


def get_confidence(row: pd.Series) -> str:
    """Label confidence from referee availability and sample size."""
    if pd.isna(row["Referee"]):
        return "LOW CONFIDENCE (no ref data)"

    if row["matches"] >= 10:
        return "HIGH CONFIDENCE"

    return "CAUTION (low sample ref)"


def add_betting_labels(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Add betting signal and confidence labels."""
    fixtures = fixtures.copy()
    fixtures["signal"] = fixtures.apply(get_signal, axis=1)
    fixtures["confidence"] = fixtures.apply(get_confidence, axis=1)
    fixtures["ultra_value"] = fixtures.apply(get_ultra_value, axis=1)
    return fixtures


def get_ultra_value(row: pd.Series) -> str:
    """Label research-backed low-card opportunities."""
    if (
        row["predicted_cards"] <= 2.7
        and row["total_cards_last5"] <= 3.0
        and row["over_4_5_prob"] < 0.30
        and row["confidence"] == "HIGH CONFIDENCE"
        and row["value_edge"] > 0.05
    ):
        return "YES"

    return "NO"


def build_outputs(fixtures: pd.DataFrame):
    """Return the full predictions table and high-confidence value bets."""
    fixtures = fixtures.copy()
    if "OddsSource" not in fixtures.columns:
        fixtures["OddsSource"] = "DEFAULT ODDS"
    if "LineupsLoaded" not in fixtures.columns:
        fixtures["LineupsLoaded"] = "NO"

    output = fixtures[OUTPUT_COLUMNS].sort_values("value_edge", ascending=False)

    top_bets = fixtures[
        ((fixtures["signal"] == "STRONG UNDER") | (fixtures["signal"] == "STRONG OVER"))
        & (fixtures["confidence"] == "HIGH CONFIDENCE")
        & (fixtures["value_edge"] > 0.05)
    ]
    top_bets = top_bets.assign(abs_edge=top_bets["edge"].abs())
    top_bets = top_bets.sort_values("abs_edge", ascending=False)

    ultra_top_bets = fixtures[fixtures["ultra_value"] == "YES"].copy()
    ultra_top_bets = ultra_top_bets.sort_values("value_edge", ascending=False)

    return output, top_bets, ultra_top_bets
