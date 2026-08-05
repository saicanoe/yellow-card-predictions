"""Betting-value calculations for verified total-match card markets."""

from math import isfinite


VERIFIED_CARD_MARKET = "verified card-market odds"
MANUAL_TEST_ODDS = "manual test odds"
UNAVAILABLE_ODDS = "unavailable odds"
VALID_ODDS_SOURCES = {
    VERIFIED_CARD_MARKET,
    MANUAL_TEST_ODDS,
    UNAVAILABLE_ODDS,
}


def _validate_probability(name: str, value: float) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number between 0 and 1.") from error
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return probability


def _validate_odds(name: str, value: float) -> float:
    try:
        odds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be decimal odds greater than 1.0.") from error
    if not isfinite(odds) or odds <= 1.0:
        raise ValueError(f"{name} must be greater than 1.0.")
    return odds


def calculate_live_betting_value(
    over_4_5_model_probability: float,
    under_4_5_model_probability: float,
    over_decimal_odds: float | None,
    under_decimal_odds: float | None,
    odds_source: str,
    stake: float = 1.0,
) -> dict:
    """Calculate two-way card-market value and return the best positive-EV side."""
    over_probability = _validate_probability(
        "over_4_5_model_probability", over_4_5_model_probability
    )
    under_probability = _validate_probability(
        "under_4_5_model_probability", under_4_5_model_probability
    )
    if odds_source not in VALID_ODDS_SOURCES:
        raise ValueError(f"odds_source must be one of: {', '.join(sorted(VALID_ODDS_SOURCES))}.")
    try:
        stake = float(stake)
    except (TypeError, ValueError) as error:
        raise ValueError("stake must be a non-negative number.") from error
    if not isfinite(stake) or stake < 0:
        raise ValueError("stake must be a non-negative number.")

    if odds_source == UNAVAILABLE_ODDS:
        if over_decimal_odds is not None or under_decimal_odds is not None:
            raise ValueError("Unavailable odds must not include prices.")
        return {
            "odds_source": UNAVAILABLE_ODDS,
            "status": "unavailable",
            "bookmaker_margin": None,
            "sides": {},
            "recommendation": None,
        }

    over_odds = _validate_odds("over_decimal_odds", over_decimal_odds)
    under_odds = _validate_odds("under_decimal_odds", under_decimal_odds)
    raw_over = 1 / over_odds
    raw_under = 1 / under_odds
    implied_total = raw_over + raw_under
    margin = implied_total - 1

    sides = {}
    for name, probability, odds, raw_implied in (
        ("Over 4.5", over_probability, over_odds, raw_over),
        ("Under 4.5", under_probability, under_odds, raw_under),
    ):
        no_vig = raw_implied / implied_total
        expected_value = probability * odds - 1
        sides[name] = {
            "model_probability": probability,
            "decimal_odds": odds,
            "raw_implied_probability": raw_implied,
            "bookmaker_margin": margin,
            "no_vig_implied_probability": no_vig,
            "probability_edge": probability - no_vig,
            "expected_value_per_unit": expected_value,
            "expected_profit": expected_value * stake,
        }

    best_side = max(sides, key=lambda side: sides[side]["expected_value_per_unit"])
    recommendation = best_side if sides[best_side]["expected_value_per_unit"] > 0 else None
    return {
        "odds_source": odds_source,
        "status": "available",
        "bookmaker_margin": margin,
        "sides": sides,
        "recommendation": recommendation,
    }
