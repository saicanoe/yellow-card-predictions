import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bet_tracker  # noqa: E402
from live_betting_value import (  # noqa: E402
    MANUAL_TEST_ODDS,
    VERIFIED_CARD_MARKET,
    calculate_live_betting_value,
)


class BetTrackerTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.tracking_path = Path(self.temp_directory.name) / "bet_tracking.csv"
        self.path_patch = patch.object(
            bet_tracker, "TRACKING_PATH", self.tracking_path
        )
        self.path_patch.start()

    def tearDown(self):
        self.path_patch.stop()
        self.temp_directory.cleanup()

    @staticmethod
    def fixture(fixture_id=123):
        return {
            "fixture_id": fixture_id,
            "date": "2026-08-10T15:00:00-04:00",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "league": "Premier League",
            "referee": "J Brooks",
        }

    @staticmethod
    def prediction(fixture_id=123):
        return {
            "fixture_id": fixture_id,
            "predicted_cards": 5.4,
            "confidence": "HIGH CONFIDENCE",
            "over_4_5_probability": 0.65,
            "under_4_5_probability": 0.35,
        }

    @staticmethod
    def value(source=VERIFIED_CARD_MARKET):
        return calculate_live_betting_value(0.65, 0.35, 2.0, 1.8, source, stake=2)

    def test_migrates_old_csv_columns_without_losing_rows(self):
        old_row = {
            "Date": "2026-01-01",
            "HomeTeam": "A",
            "AwayTeam": "B",
            "Pick": "OVER",
            "Line": 4.5,
            "Odds": 2.0,
            "Stake": 1.0,
            "Edge": 0.1,
            "Confidence": "HIGH",
            "FinalCards": pd.NA,
            "Result": "PENDING",
            "Profit": 0.0,
        }
        pd.DataFrame([old_row]).to_csv(self.tracking_path, index=False)

        migrated = bet_tracker.initialize_bet_tracking()

        self.assertEqual(len(migrated), 1)
        self.assertEqual(migrated.iloc[0]["HomeTeam"], "A")
        self.assertEqual(list(migrated.columns), bet_tracker.TRACKING_COLUMNS)
        self.assertTrue(pd.isna(migrated.iloc[0]["FixtureID"]))

    def test_saves_new_live_prediction(self):
        result = bet_tracker.save_live_prediction(
            self.fixture(), self.prediction(), self.value(), 2.0
        )

        self.assertTrue(result["saved"])
        saved = result["tracking"].iloc[0]
        self.assertEqual(saved["Pick"], "OVER 4.5")
        self.assertEqual(saved["FixtureID"], 123)
        self.assertAlmostEqual(saved["PredictedCards"], 5.4)
        self.assertAlmostEqual(saved["ModelProbability"], 0.65)
        self.assertAlmostEqual(saved["ExpectedValue"], 0.3)
        self.assertTrue(pd.isna(saved["FinalCards"]))
        self.assertEqual(saved["Result"], "PENDING")
        self.assertEqual(saved["Profit"], 0.0)

        reloaded = bet_tracker.initialize_bet_tracking().iloc[0]
        self.assertTrue(pd.isna(reloaded["FinalCards"]))
        self.assertEqual(reloaded["Result"], "PENDING")
        self.assertEqual(reloaded["Profit"], 0.0)

    def test_blank_final_cards_never_settle(self):
        tracking = pd.DataFrame(
            [
                {
                    "Pick": "OVER 4.5",
                    "Line": 4.5,
                    "Odds": 2.0,
                    "Stake": 2.0,
                    "FinalCards": pd.NA,
                    "Result": "LOSS",
                    "Profit": -2.0,
                },
                {
                    "Pick": "UNDER 4.5",
                    "Line": 4.5,
                    "Odds": 1.8,
                    "Stake": 1.0,
                    "FinalCards": -1,
                    "Result": "WIN",
                    "Profit": 0.8,
                },
            ]
        )

        evaluated = bet_tracker.evaluate_tracking(tracking)

        self.assertTrue(evaluated["FinalCards"].isna().all())
        self.assertEqual(evaluated["Result"].tolist(), ["PENDING", "PENDING"])
        self.assertEqual(evaluated["Profit"].tolist(), [0.0, 0.0])

    def test_prevents_duplicate_fixture_pick_and_line(self):
        first = bet_tracker.save_live_prediction(
            self.fixture(), self.prediction(), self.value(), 2.0
        )
        second = bet_tracker.save_live_prediction(
            self.fixture(), self.prediction(), self.value(), 3.0
        )

        self.assertTrue(first["saved"])
        self.assertFalse(second["saved"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(second["tracking"]), 1)

    def test_legacy_duplicate_falls_back_to_date_teams_and_pick(self):
        fixture = self.fixture(None)
        prediction = self.prediction(None)
        first = bet_tracker.save_live_prediction(
            fixture, prediction, self.value(), 2.0
        )
        second = bet_tracker.save_live_prediction(
            fixture, prediction, self.value(), 2.0
        )

        self.assertTrue(first["saved"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(second["tracking"]), 1)

    def test_labels_test_data(self):
        fixture = self.fixture(-1)
        prediction = self.prediction(-1)
        result = bet_tracker.save_live_prediction(
            fixture, prediction, self.value(MANUAL_TEST_ODDS), 1.0
        )

        saved = result["tracking"].iloc[0]
        self.assertTrue(saved["League"].startswith("TEST -"))
        self.assertTrue(saved["OddsSource"].startswith("TEST -"))

    def test_result_calculation_after_final_cards_entry(self):
        result = bet_tracker.save_live_prediction(
            self.fixture(), self.prediction(), self.value(), 2.0
        )
        tracking = result["tracking"]
        tracking.loc[0, "FinalCards"] = 6

        evaluated = bet_tracker.evaluate_tracking(tracking)

        self.assertEqual(evaluated.loc[0, "Result"], "WIN")
        self.assertAlmostEqual(evaluated.loc[0, "Profit"], 2.0)

    def test_profit_and_roi_calculation(self):
        tracking = pd.DataFrame(
            [
                {"Pick": "OVER 4.5", "Line": 4.5, "Odds": 2.0, "Stake": 2, "FinalCards": 6},
                {"Pick": "UNDER 4.5", "Line": 4.5, "Odds": 1.8, "Stake": 1, "FinalCards": 6},
            ]
        )

        summary = bet_tracker.calculate_summary_for_tracking(tracking)

        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertAlmostEqual(summary["total_profit"], 1.0)
        self.assertAlmostEqual(summary["roi"], 1 / 3)

    def test_existing_top_bet_logging_remains_compatible(self):
        top_bets = pd.DataFrame(
            [
                {
                    "Date": "2026-08-10",
                    "HomeTeam": "A",
                    "AwayTeam": "B",
                    "signal": "STRONG OVER",
                    "Line": 4.5,
                    "OverOdds": 2.0,
                    "value_edge": 0.1,
                    "confidence": "HIGH CONFIDENCE",
                }
            ]
        )

        tracking = bet_tracker.log_top_bets(top_bets)

        self.assertEqual(len(tracking), 1)
        self.assertEqual(tracking.iloc[0]["Pick"], "OVER 4.5")
        self.assertTrue(pd.isna(tracking.iloc[0]["FixtureID"]))


if __name__ == "__main__":
    unittest.main()
