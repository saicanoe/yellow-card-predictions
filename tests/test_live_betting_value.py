import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from live_betting_value import (  # noqa: E402
    MANUAL_TEST_ODDS,
    UNAVAILABLE_ODDS,
    VERIFIED_CARD_MARKET,
    calculate_live_betting_value,
)


class LiveBettingValueTests(unittest.TestCase):
    def calculate(self, over_probability, under_probability, over_odds, under_odds):
        return calculate_live_betting_value(
            over_probability,
            under_probability,
            over_odds,
            under_odds,
            VERIFIED_CARD_MARKET,
            stake=10,
        )

    def test_positive_over_ev(self):
        result = self.calculate(0.60, 0.40, 2.00, 1.80)
        self.assertEqual(result["recommendation"], "Over 4.5")
        self.assertAlmostEqual(result["sides"]["Over 4.5"]["expected_value_per_unit"], 0.20)
        self.assertAlmostEqual(result["sides"]["Over 4.5"]["expected_profit"], 2.00)

    def test_positive_under_ev(self):
        result = self.calculate(0.35, 0.65, 2.10, 1.80)
        self.assertEqual(result["recommendation"], "Under 4.5")

    def test_no_positive_ev(self):
        result = self.calculate(0.50, 0.50, 1.80, 1.80)
        self.assertIsNone(result["recommendation"])

    def test_bookmaker_overround_removal(self):
        result = calculate_live_betting_value(
            0.5, 0.5, 1.90, 1.90, MANUAL_TEST_ODDS
        )
        self.assertAlmostEqual(result["bookmaker_margin"], (2 / 1.9) - 1)
        self.assertAlmostEqual(
            result["sides"]["Over 4.5"]["no_vig_implied_probability"], 0.5
        )
        self.assertAlmostEqual(
            result["sides"]["Under 4.5"]["no_vig_implied_probability"], 0.5
        )

    def test_unavailable_odds_do_not_create_a_recommendation(self):
        result = calculate_live_betting_value(
            0.6, 0.4, None, None, UNAVAILABLE_ODDS
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["sides"], {})
        self.assertIsNone(result["recommendation"])

    def test_invalid_odds(self):
        for odds in (1.0, 0, -2, float("nan"), None):
            with self.subTest(odds=odds), self.assertRaises(ValueError):
                self.calculate(0.5, 0.5, odds, 1.9)

    def test_invalid_probabilities(self):
        for probability in (-0.01, 1.01, float("nan"), None):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                self.calculate(probability, 0.5, 1.9, 1.9)


if __name__ == "__main__":
    unittest.main()
