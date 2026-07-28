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

from data_loader import data_path, load_historical_matches
from features import FEATURE_COLUMNS
from odds import add_betting_labels
from xgboost import XGBClassifier, XGBRegressor

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except ImportError:
    IsotonicRegression = None
    LogisticRegression = None

BACKTEST_OUTPUT_COLUMNS = [
    "Date",
    "Season",
    "Div",
    "HomeTeam",
    "AwayTeam",
    "Referee",
    "confidence",
    "home_cards_last5",
    "away_cards_last5",
    "total_cards_last5",
    "predicted_cards",
    "over_4_5_prob",
    "under_model_prob",
    "under_book_prob",
    "MarketMode",
    "value_edge",
    "signal",
    "Pick",
    "Odds",
    "Stake",
    "total_cards",
    "Result",
    "Profit",
]

VALUE_EDGE_THRESHOLDS = [0.05, 0.08, 0.10, 0.12]
PREDICTED_CARDS_THRESHOLDS = [3.8, 3.5, 3.2]
OVER_PROB_THRESHOLDS = [0.40, 0.35, 0.30]

DIAGNOSTIC_OUTPUTS = {
    "season": "backtest_by_season.csv",
    "league": "backtest_by_league.csv",
    "referee": "backtest_by_referee.csv",
    "prediction_bucket": "backtest_by_prediction_bucket.csv",
    "probability_bucket": "backtest_by_probability_bucket.csv",
    "edge_bucket": "backtest_by_edge_bucket.csv",
    "intensity_bucket": "backtest_by_intensity_bucket.csv",
}

STRATEGY_OUTPUT = "backtest_strategy_tests.csv"
BEST_STRATEGY_ROBUSTNESS_OUTPUT = "backtest_best_strategy_robustness.csv"
BEST_STRATEGY_SENSITIVITY_OUTPUT = "backtest_best_strategy_sensitivity.csv"
CALIBRATION_OUTPUT = "backtest_calibration.csv"
CALIBRATION_COMPARISON_OUTPUT = "backtest_calibration_comparison.csv"
LOGISTIC_STRATEGY_OUTPUT = "backtest_logistic_strategy_tests.csv"
ULTRA_VALUE_OPTIMIZATION_OUTPUT = "ultra_value_optimization.csv"


def build_leakage_safe_historical(
    raw_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Build backtest features using only rows before each match."""
    df = raw_matches.copy()
    raw_rows = len(df)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["total_cards"] = df["HY"] + df["AY"]
    df["over_4_5"] = (df["total_cards"] > 4.5).astype(int)
    df["home_cards"] = df["HY"]
    df["away_cards"] = df["AY"]
    df = df.sort_values(["Date", "Div", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    # Team form and season-style averages are shifted, so the current match never
    # contributes to its own features.
    df["home_cards_last5"] = df.groupby("HomeTeam")["home_cards"].transform(
        lambda cards: cards.shift().rolling(5).mean()
    )
    df["away_cards_last5"] = df.groupby("AwayTeam")["away_cards"].transform(
        lambda cards: cards.shift().rolling(5).mean()
    )
    df["total_cards_last5"] = df["home_cards_last5"] + df["away_cards_last5"]
    df["home_season_avg"] = df.groupby("HomeTeam")["home_cards"].transform(
        lambda cards: cards.shift().expanding().mean()
    )
    df["away_season_avg"] = df.groupby("AwayTeam")["away_cards"].transform(
        lambda cards: cards.shift().expanding().mean()
    )
    df["home_advantage"] = df["home_cards_last5"] - df["away_cards_last5"]

    # Referee context is also prior-only. This replaces static referee profile
    # CSVs, which are useful live but leak future matches in a backtest.
    ref_groups = df.groupby("Referee", dropna=False)
    df["matches"] = ref_groups.cumcount()
    df["avg_total_cards"] = ref_groups["total_cards"].transform(
        lambda cards: cards.shift().expanding().mean()
    )
    df["over_4_5_rate"] = ref_groups["over_4_5"].transform(
        lambda rates: rates.shift().expanding().mean()
    )

    # Market estimates are prior-only and computed on the full chronology before
    # test slicing, so early training matches inform later simulated prices.
    df["Season"] = df["Date"].dt.year
    df["league_over_prior"] = df.groupby("Div")["over_4_5"].transform(
        lambda rates: rates.shift().expanding().mean()
    )
    df["season_over_prior"] = df.groupby(["Div", "Season"])["over_4_5"].transform(
        lambda rates: rates.shift().expanding().mean()
    )
    df["global_over_prior"] = df["over_4_5"].shift().expanding().mean()
    df["market_over_prior"] = (
        (0.65 * df["season_over_prior"]) + (0.35 * df["league_over_prior"])
    ).fillna(df["league_over_prior"])
    df["market_over_prior"] = df["market_over_prior"].fillna(df["global_over_prior"])

    team_names = pd.concat([df["HomeTeam"], df["AwayTeam"]]).dropna().unique()
    team_to_code = {team: i for i, team in enumerate(team_names)}
    league_to_code = {league: i for i, league in enumerate(sorted(df["Div"].unique()))}
    df["HomeTeamCode"] = df["HomeTeam"].map(team_to_code)
    df["AwayTeamCode"] = df["AwayTeam"].map(team_to_code)
    df["league_code"] = df["Div"].map(league_to_code)

    required = FEATURE_COLUMNS + [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "Referee",
        "HY",
        "AY",
        "total_cards",
        "over_4_5",
        "market_over_prior",
    ]
    before_drop = len(df)
    df = df.dropna(subset=required).copy()

    audit = {
        "raw_rows": raw_rows,
        "feature_rows_before_drop": before_drop,
        "feature_rows_after_drop": len(df),
        "dropped_rows": before_drop - len(df),
        "team_features": "PASS - shifted rolling/expanding team features",
        "referee_features": "PASS - shifted expanding referee features",
        "market_features": "PASS - shifted league/season market estimates",
        "full_season_averages": "PASS - no full-season future averages used",
    }
    return df.reset_index(drop=True), audit


def train_backtest_models(training_data: pd.DataFrame):
    """Train the same model family and parameters used by live predictions."""
    x_train = training_data[FEATURE_COLUMNS]

    reg_model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    reg_model.fit(x_train, training_data["total_cards"])

    clf_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )
    clf_model.fit(x_train, training_data["over_4_5"])

    return reg_model, clf_model


def _clip_probability(probability: pd.Series) -> pd.Series:
    return probability.clip(lower=0.08, upper=0.92)


def _finish_market_fields(rows: pd.DataFrame, market_mode: str) -> pd.DataFrame:
    rows["Line"] = 4.5
    rows["LineupAdjustment"] = 0.0
    rows["adjusted_cards"] = rows["predicted_cards"]
    rows["over_book_prob"] = 1 - rows["under_book_prob"]
    rows["under_model_prob"] = 1 - rows["over_4_5_prob"]
    rows["value_edge"] = rows["under_model_prob"] - rows["under_book_prob"]
    rows["value_bet"] = rows["value_edge"].apply(
        lambda edge: "YES" if edge > 0.05 else "NO"
    )
    rows["line"] = 4.5
    rows["edge"] = rows["adjusted_cards"] - rows["Line"]
    rows["MarketMode"] = market_mode
    return rows


def add_fixed_backtest_market(rows: pd.DataFrame) -> pd.DataFrame:
    """Add the original fixed 1.80 odds market for comparison."""
    rows = rows.copy()
    rows["UnderOdds"] = 1.80
    rows["OverOdds"] = 1.80
    rows["under_book_prob"] = 1 / rows["UnderOdds"]
    return _finish_market_fields(rows, "Fixed 1.80")


def add_realistic_backtest_market(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate market odds from prior league and league-season card distributions.

    This is still approximate, but it avoids granting every bet the same price.
    A small overround is added so odds behave more like bookmaker prices.
    """
    rows = rows.copy().sort_values("Date")
    estimated_over = rows["market_over_prior"].fillna(0.5)

    fair_under_prob = _clip_probability(1 - estimated_over)
    rows["under_book_prob"] = _clip_probability(fair_under_prob * 1.04)
    rows["UnderOdds"] = (1 / rows["under_book_prob"]).round(2)
    rows["OverOdds"] = (1 / _clip_probability((1 - fair_under_prob) * 1.04)).round(2)
    return _finish_market_fields(rows, "Rolling market")


def settle_bets(bets: pd.DataFrame) -> pd.DataFrame:
    """Settle 1-unit under/over 4.5 bets against actual total cards."""
    bets = bets.copy()
    bets["Pick"] = bets["signal"].apply(
        lambda signal: "UNDER 4.5" if "UNDER" in str(signal) else "OVER 4.5"
    )
    bets["Odds"] = bets.apply(
        lambda row: row["UnderOdds"] if row["Pick"] == "UNDER 4.5" else row["OverOdds"],
        axis=1,
    )
    bets["Stake"] = 1.0

    under_win = (bets["Pick"] == "UNDER 4.5") & (bets["total_cards"] <= 4)
    over_win = (bets["Pick"] == "OVER 4.5") & (bets["total_cards"] >= 5)
    bets["Result"] = "LOSS"
    bets.loc[under_win | over_win, "Result"] = "WIN"
    bets["Profit"] = bets.apply(
        lambda row: (
            row["Stake"] * (row["Odds"] - 1)
            if row["Result"] == "WIN"
            else -row["Stake"]
        ),
        axis=1,
    )
    return bets


def add_diagnostic_buckets(bets: pd.DataFrame) -> pd.DataFrame:
    """Add season and model-context buckets used for diagnostics."""
    bets = bets.copy()
    if bets.empty:
        return bets

    bets["Season"] = pd.to_datetime(bets["Date"], errors="coerce").dt.year
    bets["prediction_bucket"] = pd.cut(
        bets["predicted_cards"],
        bins=[0, 2.5, 3.0, 3.2, 3.5, 3.8, 4.5, float("inf")],
        labels=["<=2.5", "2.5-3.0", "3.0-3.2", "3.2-3.5", "3.5-3.8", "3.8-4.5", ">4.5"],
        include_lowest=True,
    )
    bets["probability_bucket"] = pd.cut(
        bets["over_4_5_prob"],
        bins=[0, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 1.0],
        labels=[
            "<=10%",
            "10-20%",
            "20-30%",
            "30-35%",
            "35-40%",
            "40-50%",
            ">50%",
        ],
        include_lowest=True,
    )
    bets["edge_bucket"] = pd.cut(
        bets["value_edge"],
        bins=[-float("inf"), 0.05, 0.08, 0.10, 0.12, 0.16, 0.20, float("inf")],
        labels=["<=5%", "5-8%", "8-10%", "10-12%", "12-16%", "16-20%", ">20%"],
        include_lowest=True,
    )
    bets["intensity_bucket"] = pd.cut(
        bets["total_cards_last5"],
        bins=[0, 3.0, 4.0, 5.0, 6.0, float("inf")],
        labels=["<=3.0", "3.0-4.0", "4.0-5.0", "5.0-6.0", ">6.0"],
        include_lowest=True,
    )
    bets["odds_bucket"] = pd.cut(
        bets["Odds"],
        bins=[0, 1.20, 1.30, 1.40, 1.50, 1.70, 2.00, float("inf")],
        labels=[
            "<=1.20",
            "1.20-1.30",
            "1.30-1.40",
            "1.40-1.50",
            "1.50-1.70",
            "1.70-2.00",
            ">2.00",
        ],
        include_lowest=True,
    )
    return bets


def summarize_bets(bets: pd.DataFrame) -> dict:
    """Calculate terminal summary metrics for settled backtest bets."""
    if bets.empty:
        return {
            "total_bets": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "avg_edge": 0.0,
            "roi_by_confidence": {},
        }

    total_staked = bets["Stake"].sum()
    total_profit = bets["Profit"].sum()
    roi_by_confidence = {}
    for confidence, group in bets.groupby("confidence", dropna=False):
        stake = group["Stake"].sum()
        profit = group["Profit"].sum()
        roi_by_confidence[str(confidence)] = profit / stake if stake else 0.0

    return {
        "total_bets": int(len(bets)),
        "win_rate": float((bets["Result"] == "WIN").mean()),
        "roi": float(total_profit / total_staked) if total_staked else 0.0,
        "avg_edge": float(bets["value_edge"].mean()),
        "roi_by_confidence": roi_by_confidence,
    }


def summarize_strategy(strategy_name: str, bets: pd.DataFrame) -> dict:
    """Calculate research strategy metrics."""
    summary = summarize_bets(bets)
    avg_profit = float(bets["Profit"].mean()) if not bets.empty else 0.0
    return {
        "strategy": strategy_name,
        "total_bets": summary["total_bets"],
        "win_rate": summary["win_rate"],
        "roi": summary["roi"],
        "avg_edge": summary["avg_edge"],
        "avg_profit": avg_profit,
    }


def summarize_group(group: pd.DataFrame) -> pd.Series:
    """Calculate grouped diagnostic metrics for settled bets."""
    total_bets = len(group)
    total_staked = group["Stake"].sum()
    total_profit = group["Profit"].sum()
    return pd.Series(
        {
            "total_bets": int(total_bets),
            "win_rate": float((group["Result"] == "WIN").mean()) if total_bets else 0.0,
            "roi": float(total_profit / total_staked) if total_staked else 0.0,
            "avg_profit": float(group["Profit"].mean()) if total_bets else 0.0,
            "avg_edge": float(group["value_edge"].mean()) if total_bets else 0.0,
        }
    )


def build_group_diagnostic(bets: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Build one diagnostic table for a grouping column."""
    if bets.empty or group_column not in bets.columns:
        return pd.DataFrame(
            columns=[
                group_column,
                "total_bets",
                "win_rate",
                "roi",
                "avg_profit",
                "avg_edge",
            ]
        )

    diagnostic = (
        bets.groupby(group_column, dropna=False, observed=True)
        .apply(summarize_group)
        .reset_index()
    )
    diagnostic = diagnostic[diagnostic["total_bets"] > 0].copy()
    diagnostic = diagnostic.sort_values(
        ["roi", "total_bets"], ascending=[False, False]
    ).reset_index(drop=True)
    return diagnostic


def save_diagnostics(bets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Save diagnostics grouped by season, league, referee, and model buckets."""
    bets = add_diagnostic_buckets(bets)
    diagnostics = {
        "season": build_group_diagnostic(bets, "Season"),
        "league": build_group_diagnostic(bets, "Div"),
        "referee": build_group_diagnostic(bets, "Referee"),
        "prediction_bucket": build_group_diagnostic(bets, "prediction_bucket"),
        "probability_bucket": build_group_diagnostic(bets, "probability_bucket"),
        "edge_bucket": build_group_diagnostic(bets, "edge_bucket"),
        "intensity_bucket": build_group_diagnostic(bets, "intensity_bucket"),
    }

    for key, filename in DIAGNOSTIC_OUTPUTS.items():
        diagnostics[key].to_csv(data_path(filename), index=False)

    return diagnostics


def price_and_label_predictions(
    predicted: pd.DataFrame, market_mode: str
) -> pd.DataFrame:
    """Add market pricing and betting labels without applying selection filters."""
    if predicted.empty:
        return pd.DataFrame()

    if market_mode == "fixed":
        priced = add_fixed_backtest_market(predicted)
    elif market_mode == "realistic":
        priced = add_realistic_backtest_market(predicted)
    else:
        raise ValueError(f"Unknown market mode: {market_mode}")

    return add_betting_labels(priced)


def select_research_bets(
    priced: pd.DataFrame,
    value_edge_threshold: float,
    predicted_cards_threshold: float,
    over_prob_threshold: float,
) -> pd.DataFrame:
    """Select stricter UNDER candidates for threshold research."""
    return priced[
        (priced["signal"] == "STRONG UNDER")
        & (priced["confidence"] == "HIGH CONFIDENCE")
        & (priced["value_edge"] > value_edge_threshold)
        & (priced["predicted_cards"] < predicted_cards_threshold)
        & (priced["over_4_5_prob"] < over_prob_threshold)
    ].copy()


def run_threshold_experiments(priced: pd.DataFrame) -> pd.DataFrame:
    """Evaluate value, card total, and probability threshold combinations."""
    rows = []
    for value_edge_threshold in VALUE_EDGE_THRESHOLDS:
        for predicted_cards_threshold in PREDICTED_CARDS_THRESHOLDS:
            for over_prob_threshold in OVER_PROB_THRESHOLDS:
                candidates = select_research_bets(
                    priced,
                    value_edge_threshold,
                    predicted_cards_threshold,
                    over_prob_threshold,
                )
                bets = settle_bets(candidates)
                summary = summarize_bets(bets)
                rows.append(
                    {
                        "value_edge_gt": value_edge_threshold,
                        "predicted_cards_lt": predicted_cards_threshold,
                        "over_4_5_prob_lt": over_prob_threshold,
                        "total_bets": summary["total_bets"],
                        "win_rate": summary["win_rate"],
                        "roi": summary["roi"],
                        "avg_edge": summary["avg_edge"],
                    }
                )

    threshold_results = pd.DataFrame(rows)
    threshold_results = threshold_results.sort_values(
        ["roi", "total_bets", "avg_edge"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    threshold_results.to_csv(data_path("backtest_thresholds.csv"), index=False)
    return threshold_results


def baseline_research_candidates(priced: pd.DataFrame) -> pd.DataFrame:
    """Return the current research baseline candidate set before settlement."""
    return priced[
        priced["signal"].isin(["STRONG UNDER", "STRONG OVER"])
        & (priced["confidence"] == "HIGH CONFIDENCE")
        & (priced["value_edge"] > 0.05)
    ].copy()


def run_strategy_tests(priced: pd.DataFrame) -> pd.DataFrame:
    """Evaluate research-only filtered strategy variants."""
    baseline = baseline_research_candidates(priced)
    strategies = [
        ("Current baseline strategy", baseline),
        (
            "predicted_cards <= 2.5 only",
            baseline[baseline["predicted_cards"] <= 2.5],
        ),
        (
            "total_cards_last5 <= 3.0 only",
            baseline[baseline["total_cards_last5"] <= 3.0],
        ),
        (
            "predicted_cards <= 2.5 AND total_cards_last5 <= 3.0",
            baseline[
                (baseline["predicted_cards"] <= 2.5)
                & (baseline["total_cards_last5"] <= 3.0)
            ],
        ),
        (
            "Exclude total_cards_last5 > 6.0",
            baseline[baseline["total_cards_last5"] <= 6.0],
        ),
        (
            "predicted_cards <= 2.5 AND exclude total_cards_last5 > 6.0",
            baseline[
                (baseline["predicted_cards"] <= 2.5)
                & (baseline["total_cards_last5"] <= 6.0)
            ],
        ),
        (
            "value_edge > 0.10 AND predicted_cards <= 2.5",
            baseline[
                (baseline["value_edge"] > 0.10) & (baseline["predicted_cards"] <= 2.5)
            ],
        ),
        (
            "value_edge > 0.10 AND predicted_cards <= 2.5 AND total_cards_last5 <= 3.0",
            baseline[
                (baseline["value_edge"] > 0.10)
                & (baseline["predicted_cards"] <= 2.5)
                & (baseline["total_cards_last5"] <= 3.0)
            ],
        ),
    ]

    rows = []
    for strategy_name, candidates in strategies:
        rows.append(summarize_strategy(strategy_name, settle_bets(candidates)))

    strategy_results = pd.DataFrame(rows)
    strategy_results = strategy_results.sort_values(
        ["roi", "total_bets"], ascending=[False, False]
    ).reset_index(drop=True)
    strategy_results.to_csv(data_path(STRATEGY_OUTPUT), index=False)
    return strategy_results


def ultra_value_candidates(priced: pd.DataFrame) -> pd.DataFrame:
    """Return the current live Ultra Value rule using raw model probabilities."""
    return priced[
        (priced["signal"] == "STRONG UNDER")
        & (priced["confidence"] == "HIGH CONFIDENCE")
        & (priced["value_edge"] > 0.05)
        & (priced["predicted_cards"] <= 2.7)
        & (priced["total_cards_last5"] <= 3.0)
    ].copy()


def run_logistic_strategy_tests(priced: pd.DataFrame) -> pd.DataFrame:
    """Test logistic-calibrated under probability as a research-only selector."""
    if priced.empty:
        results = pd.DataFrame(
            columns=[
                "strategy",
                "logistic_under_prob_gt",
                "total_bets",
                "win_rate",
                "roi",
                "avg_edge",
                "avg_profit",
            ]
        )
        results.to_csv(data_path(LOGISTIC_STRATEGY_OUTPUT), index=False)
        return results

    base_filter = (
        (priced["signal"] == "STRONG UNDER")
        & (priced["confidence"] == "HIGH CONFIDENCE")
        & (priced["value_edge"] > 0.05)
        & (priced["predicted_cards"] <= 2.7)
        & (priced["total_cards_last5"] <= 3.0)
    )

    rows = []
    ultra_bets = settle_bets(ultra_value_candidates(priced))
    ultra_summary = summarize_strategy("Current Ultra Value strategy", ultra_bets)
    ultra_summary["logistic_under_prob_gt"] = None
    rows.append(ultra_summary)

    logistic_column = "over_4_5_prob_logistic_calibrated"
    if logistic_column in priced.columns:
        logistic_under_prob = 1 - priced[logistic_column]
        for threshold in [0.70, 0.75, 0.80]:
            candidates = priced[base_filter & (logistic_under_prob > threshold)].copy()
            bets = settle_bets(candidates)
            summary = summarize_strategy(
                f"Logistic under prob > {threshold:.2f}", bets
            )
            summary["logistic_under_prob_gt"] = threshold
            rows.append(summary)

    results = pd.DataFrame(rows)
    results = results[
        [
            "strategy",
            "logistic_under_prob_gt",
            "total_bets",
            "win_rate",
            "roi",
            "avg_edge",
            "avg_profit",
        ]
    ]
    results = results.sort_values(["roi", "total_bets"], ascending=[False, False])
    results.to_csv(data_path(LOGISTIC_STRATEGY_OUTPUT), index=False)
    return results


def _roi_by_group_text(bets: pd.DataFrame, group_column: str) -> tuple[str, float, float]:
    """Return readable group ROI detail plus minimum ROI and positive-group rate."""
    if bets.empty or group_column not in bets.columns:
        return "", 0.0, 0.0

    rows = []
    roi_values = []
    for group_name, group in bets.groupby(group_column, dropna=False):
        stake = group["Stake"].sum()
        roi = float(group["Profit"].sum() / stake) if stake else 0.0
        roi_values.append(roi)
        rows.append(f"{group_name}:{len(group)} bets/{roi:.1%}")

    positive_rate = (
        sum(1 for roi in roi_values if roi > 0) / len(roi_values) if roi_values else 0.0
    )
    return "; ".join(rows), min(roi_values) if roi_values else 0.0, positive_rate


def _roi_stability_std(
    bets: pd.DataFrame, group_columns: list[str], min_group_bets: int = 5
) -> float:
    """Measure ROI spread across season/referee/league groups."""
    roi_values = []
    for group_column in group_columns:
        if bets.empty or group_column not in bets.columns:
            continue
        for _, group in bets.groupby(group_column, dropna=False):
            if len(group) < min_group_bets:
                continue
            stake = group["Stake"].sum()
            if stake:
                roi_values.append(float(group["Profit"].sum() / stake))
    return float(pd.Series(roi_values).std(ddof=0)) if roi_values else 0.0


def run_ultra_value_optimization(priced: pd.DataFrame) -> pd.DataFrame:
    """Grid-search Ultra Value variants without changing the live rule."""
    rows = []
    for predicted_cards_threshold in [2.5, 2.6, 2.7, 2.8]:
        for intensity_threshold in [2.5, 3.0, 3.5]:
            for value_edge_threshold in [0.05, 0.08, 0.10, 0.12]:
                for over_prob_threshold in [0.30, 0.35, 0.40]:
                    candidates = priced[
                        (priced["signal"] == "STRONG UNDER")
                        & (priced["confidence"] == "HIGH CONFIDENCE")
                        & (priced["predicted_cards"] <= predicted_cards_threshold)
                        & (priced["total_cards_last5"] <= intensity_threshold)
                        & (priced["value_edge"] > value_edge_threshold)
                        & (priced["over_4_5_prob"] < over_prob_threshold)
                    ].copy()
                    bets = settle_bets(candidates)
                    summary = summarize_bets(bets)

                    season_roi, season_min_roi, season_positive_rate = (
                        _roi_by_group_text(bets, "Season")
                    )
                    referee_roi, referee_min_roi, referee_positive_rate = (
                        _roi_by_group_text(bets, "Referee")
                    )
                    league_roi, league_min_roi, league_positive_rate = (
                        _roi_by_group_text(bets, "Div")
                    )
                    stability_std = _roi_stability_std(
                        bets, ["Season", "Referee", "Div"]
                    )

                    rows.append(
                        {
                            "predicted_cards_lte": predicted_cards_threshold,
                            "total_cards_last5_lte": intensity_threshold,
                            "value_edge_gt": value_edge_threshold,
                            "over_4_5_prob_lt": over_prob_threshold,
                            "total_bets": summary["total_bets"],
                            "minimum_40_bets": (
                                "YES" if summary["total_bets"] >= 40 else "NO"
                            ),
                            "win_rate": summary["win_rate"],
                            "roi": summary["roi"],
                            "avg_edge": summary["avg_edge"],
                            "avg_profit": (
                                float(bets["Profit"].mean())
                                if not bets.empty
                                else 0.0
                            ),
                            "season_min_roi": season_min_roi,
                            "referee_min_roi": referee_min_roi,
                            "league_min_roi": league_min_roi,
                            "season_positive_roi_rate": season_positive_rate,
                            "referee_positive_roi_rate": referee_positive_rate,
                            "league_positive_roi_rate": league_positive_rate,
                            "roi_stability_std": stability_std,
                            "stability_score": summary["roi"] - stability_std,
                            "roi_by_season": season_roi,
                            "roi_by_referee": referee_roi,
                            "roi_by_league": league_roi,
                        }
                    )

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["roi", "total_bets", "stability_score"], ascending=[False, False, False]
    ).reset_index(drop=True)
    results.to_csv(data_path(ULTRA_VALUE_OPTIMIZATION_OUTPUT), index=False)
    return results


def best_strategy_candidates(priced: pd.DataFrame) -> pd.DataFrame:
    """Return the current best research-only filtered strategy candidates."""
    baseline = baseline_research_candidates(priced)
    return baseline[
        (baseline["predicted_cards"] <= 2.5) & (baseline["total_cards_last5"] <= 3.0)
    ].copy()


def add_sample_warning(rows: pd.DataFrame) -> pd.DataFrame:
    """Flag grouped robustness rows with fewer than 20 bets."""
    rows = rows.copy()
    rows["sample_warning"] = rows["total_bets"].apply(
        lambda total_bets: "LOW SAMPLE" if total_bets < 20 else ""
    )
    return rows


def run_best_strategy_robustness(priced: pd.DataFrame) -> pd.DataFrame:
    """Evaluate best filtered strategy by key stability dimensions."""
    bets = add_diagnostic_buckets(settle_bets(best_strategy_candidates(priced)))
    groups = {
        "season": "Season",
        "league": "Div",
        "referee": "Referee",
        "odds_bucket": "odds_bucket",
        "prediction_bucket": "prediction_bucket",
    }

    diagnostics = []
    for dimension, column in groups.items():
        diagnostic = build_group_diagnostic(bets, column)
        if diagnostic.empty:
            continue
        diagnostic = diagnostic.rename(columns={column: "group"})
        diagnostic.insert(0, "dimension", dimension)
        diagnostics.append(add_sample_warning(diagnostic))

    if diagnostics:
        robustness = pd.concat(diagnostics, ignore_index=True)
    else:
        robustness = pd.DataFrame(
            columns=[
                "dimension",
                "group",
                "total_bets",
                "win_rate",
                "roi",
                "avg_profit",
                "avg_edge",
                "sample_warning",
            ]
        )

    robustness.to_csv(data_path(BEST_STRATEGY_ROBUSTNESS_OUTPUT), index=False)
    return robustness


def run_best_strategy_sensitivity(priced: pd.DataFrame) -> pd.DataFrame:
    """Test nearby thresholds around the best filtered strategy."""
    baseline = baseline_research_candidates(priced)
    rows = []
    for predicted_cards_threshold in [2.4, 2.5, 2.6, 2.7]:
        for intensity_threshold in [2.5, 3.0, 3.5, 4.0]:
            candidates = baseline[
                (baseline["predicted_cards"] <= predicted_cards_threshold)
                & (baseline["total_cards_last5"] <= intensity_threshold)
            ].copy()
            bets = settle_bets(candidates)
            summary = summarize_bets(bets)
            rows.append(
                {
                    "predicted_cards_lte": predicted_cards_threshold,
                    "total_cards_last5_lte": intensity_threshold,
                    "total_bets": summary["total_bets"],
                    "win_rate": summary["win_rate"],
                    "roi": summary["roi"],
                    "avg_edge": summary["avg_edge"],
                    "avg_profit": (
                        float(bets["Profit"].mean()) if not bets.empty else 0.0
                    ),
                    "sample_warning": (
                        "LOW SAMPLE" if summary["total_bets"] < 20 else ""
                    ),
                }
            )

    sensitivity = pd.DataFrame(rows).sort_values(
        ["roi", "total_bets"], ascending=[False, False]
    )
    sensitivity.to_csv(data_path(BEST_STRATEGY_SENSITIVITY_OUTPUT), index=False)
    return sensitivity


def run_calibration_diagnostics(predicted: pd.DataFrame) -> pd.DataFrame:
    """Bucket over-4.5 probabilities and compare them to actual hit rates."""
    calibration = build_calibration_table(predicted, "over_4_5_prob")
    calibration.to_csv(data_path(CALIBRATION_OUTPUT), index=False)
    return calibration


def build_calibration_table(
    predicted: pd.DataFrame, probability_column: str
) -> pd.DataFrame:
    """Create 10 probability buckets for a raw or calibrated probability column."""
    if predicted.empty or probability_column not in predicted.columns:
        calibration = pd.DataFrame(
            columns=[
                "probability_bucket",
                "count",
                "avg_predicted_prob",
                "actual_over_4_5_rate",
                "calibration_error",
            ]
        )
        return calibration

    rows = predicted.copy()
    rows["probability_bucket"] = pd.cut(
        rows[probability_column],
        bins=[i / 10 for i in range(11)],
        labels=[f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)],
        include_lowest=True,
        right=False,
    )
    rows.loc[rows[probability_column] == 1.0, "probability_bucket"] = "0.9-1.0"

    calibration = (
        rows.groupby("probability_bucket", observed=False)
        .agg(
            count=("over_4_5", "size"),
            avg_predicted_prob=(probability_column, "mean"),
            actual_over_4_5_rate=("over_4_5", "mean"),
        )
        .reset_index()
    )
    calibration["calibration_error"] = (
        calibration["avg_predicted_prob"] - calibration["actual_over_4_5_rate"]
    ).abs()
    return calibration


def summarize_calibration_error(predicted: pd.DataFrame, probability_column: str) -> dict:
    """Return headline calibration error metrics for one probability column."""
    calibration = build_calibration_table(predicted, probability_column)
    non_empty = calibration[calibration["count"] > 0].copy()
    if non_empty.empty:
        return {
            "mean_abs_calibration_error": 0.0,
            "worst_bucket": "",
            "worst_bucket_error": 0.0,
        }

    worst_bucket = non_empty.sort_values("calibration_error", ascending=False).iloc[0]
    return {
        "mean_abs_calibration_error": float(non_empty["calibration_error"].mean()),
        "worst_bucket": str(worst_bucket["probability_bucket"]),
        "worst_bucket_error": float(worst_bucket["calibration_error"]),
    }


def apply_walk_forward_calibration(
    test_rows: pd.DataFrame, calibration_history: pd.DataFrame
) -> pd.DataFrame:
    """
    Calibrate current block probabilities using only earlier out-of-sample blocks.

    The first blocks intentionally fall back to raw probabilities until there is
    enough prior prediction history with both classes represented.
    """
    calibrated = test_rows.copy()
    raw_prob = calibrated["over_4_5_prob"].clip(0, 1)
    calibrated["over_4_5_prob_logistic_calibrated"] = raw_prob
    calibrated["over_4_5_prob_isotonic_calibrated"] = raw_prob
    calibrated["calibration_history_size"] = len(calibration_history)

    has_enough_history = (
        len(calibration_history) >= 100 and calibration_history["over_4_5"].nunique() == 2
    )
    if not has_enough_history:
        return calibrated

    history_prob = calibration_history["over_4_5_prob"].clip(0, 1)
    history_target = calibration_history["over_4_5"].astype(int)

    if LogisticRegression is not None:
        logistic = LogisticRegression(solver="lbfgs")
        logistic.fit(history_prob.to_numpy().reshape(-1, 1), history_target)
        calibrated["over_4_5_prob_logistic_calibrated"] = logistic.predict_proba(
            raw_prob.to_numpy().reshape(-1, 1)
        )[:, 1]

    if IsotonicRegression is not None:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(history_prob, history_target)
        calibrated["over_4_5_prob_isotonic_calibrated"] = isotonic.predict(raw_prob)

    calibrated["over_4_5_prob_logistic_calibrated"] = calibrated[
        "over_4_5_prob_logistic_calibrated"
    ].clip(0, 1)
    calibrated["over_4_5_prob_isotonic_calibrated"] = calibrated[
        "over_4_5_prob_isotonic_calibrated"
    ].clip(0, 1)
    return calibrated


def run_calibration_comparison(predicted: pd.DataFrame) -> pd.DataFrame:
    """Compare raw and walk-forward calibrated probabilities on error and ROI."""
    methods = [
        ("raw", "over_4_5_prob", "available"),
        (
            "logistic",
            "over_4_5_prob_logistic_calibrated",
            "available" if LogisticRegression is not None else "sklearn unavailable",
        ),
        (
            "isotonic",
            "over_4_5_prob_isotonic_calibrated",
            "available" if IsotonicRegression is not None else "sklearn unavailable",
        ),
    ]

    rows = []
    for method, probability_column, status in methods:
        if predicted.empty or probability_column not in predicted.columns:
            summary = summarize_bets(pd.DataFrame())
            calibration_summary = summarize_calibration_error(
                pd.DataFrame(), probability_column
            )
        else:
            method_predictions = predicted.copy()
            method_predictions["over_4_5_prob"] = method_predictions[
                probability_column
            ]
            _, summary = run_market_backtest(method_predictions, "realistic")
            calibration_summary = summarize_calibration_error(
                method_predictions, "over_4_5_prob"
            )

        rows.append(
            {
                "method": method,
                "status": status,
                "mean_abs_calibration_error": calibration_summary[
                    "mean_abs_calibration_error"
                ],
                "worst_bucket": calibration_summary["worst_bucket"],
                "worst_bucket_error": calibration_summary["worst_bucket_error"],
                "roi": summary["roi"],
                "total_bets": summary["total_bets"],
                "win_rate": summary["win_rate"],
                "avg_edge": summary["avg_edge"],
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(data_path(CALIBRATION_COMPARISON_OUTPUT), index=False)
    return comparison


def evaluate_best_strategy_verdict(
    robustness: pd.DataFrame, sensitivity: pd.DataFrame
) -> str:
    """Classify stability of the best filtered strategy."""
    target = sensitivity[
        (sensitivity["predicted_cards_lte"] == 2.5)
        & (sensitivity["total_cards_last5_lte"] == 3.0)
    ]
    target_bets = int(target["total_bets"].iloc[0]) if not target.empty else 0
    target_roi = float(target["roi"].iloc[0]) if not target.empty else 0.0

    nearby = sensitivity[
        (sensitivity["predicted_cards_lte"].between(2.4, 2.6))
        & (sensitivity["total_cards_last5_lte"].between(2.5, 3.5))
    ]
    positive_nearby = int((nearby["roi"] > 0).sum()) if not nearby.empty else 0
    enough_nearby = int((nearby["total_bets"] >= 20).sum()) if not nearby.empty else 0

    season_rows = robustness[robustness["dimension"] == "season"]
    enough_seasons = season_rows[season_rows["total_bets"] >= 20]
    positive_enough_seasons = (
        int((enough_seasons["roi"] > 0).sum()) if not enough_seasons.empty else 0
    )

    if (
        target_bets >= 50
        and target_roi > 0
        and positive_nearby >= max(5, len(nearby) // 2)
        and positive_enough_seasons >= 2
    ):
        return "ROBUST"

    if target_roi > 0 and target_bets >= 20:
        return "PROMISING BUT LOW SAMPLE"

    return "NOT ROBUST"


def generate_chronological_predictions(
    min_train_matches: int = 500, test_block_size: int = 100
) -> tuple[pd.DataFrame, dict]:
    """
    Run a chronological backtest using historical features.

    The script trains on earlier matches and predicts later blocks. This keeps the
    live prediction code unchanged while giving an MVP profitability read.
    """
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

    raw_matches = load_historical_matches()
    historical, audit = build_leakage_safe_historical(raw_matches)
    historical = historical.sort_values("Date").reset_index(drop=True)

    predictions = []
    calibration_history = pd.DataFrame(columns=["over_4_5_prob", "over_4_5"])
    for start in range(min_train_matches, len(historical), test_block_size):
        end = min(start + test_block_size, len(historical))
        train_rows = historical.iloc[:start]
        test_rows = historical.iloc[start:end].copy()

        if train_rows["over_4_5"].nunique() < 2:
            continue

        reg_model, clf_model = train_backtest_models(train_rows)
        test_rows["predicted_cards"] = reg_model.predict(test_rows[FEATURE_COLUMNS])
        test_rows["over_4_5_prob"] = clf_model.predict_proba(
            test_rows[FEATURE_COLUMNS]
        )[:, 1]
        test_rows = apply_walk_forward_calibration(test_rows, calibration_history)
        predictions.append(test_rows)
        calibration_history = pd.concat(
            [
                calibration_history,
                test_rows[["over_4_5_prob", "over_4_5"]],
            ],
            ignore_index=True,
        )

    if not predictions:
        return pd.DataFrame(), audit

    return pd.concat(predictions, ignore_index=True), audit


def run_market_backtest(predicted: pd.DataFrame, market_mode: str):
    """Apply one market pricing method, filter bets, settle, and summarize."""
    if predicted.empty:
        results = pd.DataFrame(columns=BACKTEST_OUTPUT_COLUMNS)
        return results, summarize_bets(results)

    priced = price_and_label_predictions(predicted, market_mode)

    candidates = priced[
        priced["signal"].isin(["STRONG UNDER", "STRONG OVER"])
        & (priced["confidence"] == "HIGH CONFIDENCE")
        & (priced["value_edge"] > 0.05)
    ].copy()

    bets = settle_bets(candidates)
    bets = add_diagnostic_buckets(bets)
    results = bets[BACKTEST_OUTPUT_COLUMNS].copy()
    return results, summarize_bets(results)


def run_backtest(min_train_matches: int = 500, test_block_size: int = 100):
    """
    Run fixed-odds and realistic-market backtests from the same predictions.

    The model simulation remains chronological. The original fixed 1.80 version
    is preserved as a baseline, while the main output uses rolling market odds.
    """
    predicted, audit = generate_chronological_predictions(
        min_train_matches, test_block_size
    )

    fixed_results, fixed_summary = run_market_backtest(predicted, "fixed")
    realistic_results, realistic_summary = run_market_backtest(predicted, "realistic")
    diagnostics = save_diagnostics(realistic_results)
    realistic_priced = price_and_label_predictions(predicted, "realistic")
    threshold_results = run_threshold_experiments(realistic_priced)
    strategy_results = run_strategy_tests(realistic_priced)
    logistic_strategy_results = run_logistic_strategy_tests(realistic_priced)
    ultra_optimization_results = run_ultra_value_optimization(realistic_priced)
    robustness_results = run_best_strategy_robustness(realistic_priced)
    sensitivity_results = run_best_strategy_sensitivity(realistic_priced)
    calibration_results = run_calibration_diagnostics(predicted)
    calibration_comparison = run_calibration_comparison(predicted)
    robustness_verdict = evaluate_best_strategy_verdict(
        robustness_results, sensitivity_results
    )

    fixed_results.to_csv(data_path("backtest_results_fixed_odds.csv"), index=False)
    realistic_results.to_csv(data_path("backtest_results.csv"), index=False)
    return (
        realistic_results,
        realistic_summary,
        fixed_summary,
        audit,
        threshold_results,
        diagnostics,
        strategy_results,
        logistic_strategy_results,
        ultra_optimization_results,
        robustness_results,
        sensitivity_results,
        calibration_results,
        calibration_comparison,
        robustness_verdict,
    )


def print_leakage_audit(audit: dict):
    """Print checks showing the backtest uses prior-only information."""
    print("\nLEAKAGE AUDIT")
    print(f"Raw historical rows: {audit['raw_rows']}")
    print(f"Rows after prior-only feature drop: {audit['feature_rows_after_drop']}")
    print(f"Rows dropped for insufficient prior history: {audit['dropped_rows']}")
    print(f"Team features: {audit['team_features']}")
    print(f"Referee averages: {audit['referee_features']}")
    print(f"League/season market odds: {audit['market_features']}")
    print(f"Full-season averages: {audit['full_season_averages']}")


def print_summary(realistic_summary: dict, fixed_summary: dict, audit: dict):
    """Print a compact terminal summary."""
    print_leakage_audit(audit)

    print("\nBACKTEST SUMMARY")
    print("Market mode: Rolling league/season market odds")
    print(f"Total bets: {realistic_summary['total_bets']}")
    print(f"Win rate: {realistic_summary['win_rate']:.1%}")
    print(f"ROI realistic odds: {realistic_summary['roi']:.1%}")
    print(f"Avg edge realistic odds: {realistic_summary['avg_edge']:.1%}")

    print("\nCOMPARISON")
    print(f"Previous fixed-odds ROI: {fixed_summary['roi']:.1%}")
    print(f"Fixed-odds total bets: {fixed_summary['total_bets']}")
    print("ROI difference: " f"{realistic_summary['roi'] - fixed_summary['roi']:+.1%}")

    print("\nROI BY CONFIDENCE")
    if realistic_summary["roi_by_confidence"]:
        for confidence, roi in realistic_summary["roi_by_confidence"].items():
            print(f"{confidence}: {roi:.1%}")
    else:
        print("No settled bets")

    print("\nSaved:")
    print("data/backtest_results.csv")
    print("data/backtest_results_fixed_odds.csv")
    print("data/backtest_thresholds.csv")
    print(f"data/{STRATEGY_OUTPUT}")
    print(f"data/{BEST_STRATEGY_ROBUSTNESS_OUTPUT}")
    print(f"data/{BEST_STRATEGY_SENSITIVITY_OUTPUT}")
    print(f"data/{CALIBRATION_OUTPUT}")
    print(f"data/{CALIBRATION_COMPARISON_OUTPUT}")
    print(f"data/{LOGISTIC_STRATEGY_OUTPUT}")
    print(f"data/{ULTRA_VALUE_OPTIMIZATION_OUTPUT}")
    for filename in DIAGNOSTIC_OUTPUTS.values():
        print(f"data/{filename}")


def print_threshold_table(threshold_results: pd.DataFrame):
    """Print threshold experiment comparison table."""
    print("\nTHRESHOLD EXPERIMENTS")
    if threshold_results.empty:
        print("No threshold results")
        return

    display = threshold_results.copy()
    display["win_rate"] = display["win_rate"].map(lambda value: f"{value:.1%}")
    display["roi"] = display["roi"].map(lambda value: f"{value:.1%}")
    display["avg_edge"] = display["avg_edge"].map(lambda value: f"{value:.1%}")
    print(display.to_string(index=False))


def print_strategy_table(strategy_results: pd.DataFrame):
    """Print research-only filtered strategy comparison."""
    print("\nFILTERED STRATEGY TESTS")
    if strategy_results.empty:
        print("No strategy results")
        return

    display = strategy_results.copy()
    display["win_rate"] = display["win_rate"].map(lambda value: f"{value:.1%}")
    display["roi"] = display["roi"].map(lambda value: f"{value:.1%}")
    display["avg_edge"] = display["avg_edge"].map(lambda value: f"{value:.1%}")
    display["avg_profit"] = display["avg_profit"].map(lambda value: f"{value:.2f}")
    print(display.to_string(index=False))


def print_logistic_strategy_table(logistic_strategy_results: pd.DataFrame):
    """Print logistic calibration selector experiments against Ultra Value."""
    print("\nLOGISTIC STRATEGY TESTS")
    if logistic_strategy_results.empty:
        print("No logistic strategy results")
        return

    display = logistic_strategy_results.copy()
    display["logistic_under_prob_gt"] = display["logistic_under_prob_gt"].apply(
        lambda value: "" if pd.isna(value) else f"{value:.2f}"
    )
    for column in ["win_rate", "roi", "avg_edge"]:
        display[column] = display[column].map(lambda value: f"{value:.1%}")
    display["avg_profit"] = display["avg_profit"].map(lambda value: f"{value:.2f}")
    print(display.to_string(index=False))

    ultra = logistic_strategy_results[
        logistic_strategy_results["strategy"] == "Current Ultra Value strategy"
    ]
    logistic = logistic_strategy_results[
        logistic_strategy_results["strategy"] != "Current Ultra Value strategy"
    ]
    if ultra.empty or logistic.empty:
        return

    best_logistic = logistic.sort_values(
        ["roi", "total_bets"], ascending=[False, False]
    ).iloc[0]
    ultra_row = ultra.iloc[0]
    print(
        "\nBest logistic vs Ultra Value: "
        f"{best_logistic['strategy']} "
        f"ROI {best_logistic['roi']:.1%} on {int(best_logistic['total_bets'])} bets "
        "vs Ultra Value "
        f"ROI {ultra_row['roi']:.1%} on {int(ultra_row['total_bets'])} bets"
    )


def _format_ultra_strategy(row: pd.Series) -> str:
    return (
        f"predicted_cards <= {row['predicted_cards_lte']:.1f}, "
        f"total_cards_last5 <= {row['total_cards_last5_lte']:.1f}, "
        f"value_edge > {row['value_edge_gt']:.2f}, "
        f"over_4_5_prob < {row['over_4_5_prob_lt']:.2f}"
    )


def print_ultra_value_optimization(ultra_optimization_results: pd.DataFrame):
    """Print the strongest and steadiest Ultra Value research variants."""
    print("\nULTRA VALUE OPTIMIZATION")
    if ultra_optimization_results.empty:
        print("No Ultra Value optimization results")
        return

    best_roi = ultra_optimization_results.sort_values(
        ["roi", "total_bets"], ascending=[False, False]
    ).iloc[0]
    minimum_40 = ultra_optimization_results[
        ultra_optimization_results["total_bets"] >= 40
    ].copy()
    best_40 = (
        minimum_40.sort_values(["roi", "total_bets"], ascending=[False, False]).iloc[0]
        if not minimum_40.empty
        else None
    )
    safest_pool = minimum_40[minimum_40["roi"] > 0].copy()
    if safest_pool.empty:
        safest_pool = minimum_40 if not minimum_40.empty else ultra_optimization_results
    safest = safest_pool.sort_values(
        ["stability_score", "roi", "total_bets"], ascending=[False, False, False]
    ).iloc[0]

    print(
        "Best by ROI: "
        f"{_format_ultra_strategy(best_roi)} "
        f"-> ROI {best_roi['roi']:.1%}, "
        f"{int(best_roi['total_bets'])} bets, "
        f"win rate {best_roi['win_rate']:.1%}"
    )
    if best_40 is None:
        print("Best with at least 40 bets: no strategy reached 40 bets")
    else:
        print(
            "Best with at least 40 bets: "
            f"{_format_ultra_strategy(best_40)} "
            f"-> ROI {best_40['roi']:.1%}, "
            f"{int(best_40['total_bets'])} bets, "
            f"win rate {best_40['win_rate']:.1%}"
        )
    print(
        "Safest by ROI stability: "
        f"{_format_ultra_strategy(safest)} "
        f"-> stability score {safest['stability_score']:.1%}, "
        f"ROI {safest['roi']:.1%}, "
        f"{int(safest['total_bets'])} bets"
    )


def print_best_strategy_robustness(
    robustness_results: pd.DataFrame,
    sensitivity_results: pd.DataFrame,
    robustness_verdict: str,
):
    """Print best-strategy stability checks and verdict."""
    print("\nBEST STRATEGY ROBUSTNESS")
    print("Strategy: predicted_cards <= 2.5 AND total_cards_last5 <= 3.0")
    print(f"Verdict: {robustness_verdict}")

    if robustness_results.empty:
        print("No robustness groups")
    else:
        display = robustness_results.copy()
        display["win_rate"] = display["win_rate"].map(lambda value: f"{value:.1%}")
        display["roi"] = display["roi"].map(lambda value: f"{value:.1%}")
        display["avg_edge"] = display["avg_edge"].map(lambda value: f"{value:.1%}")
        display["avg_profit"] = display["avg_profit"].map(lambda value: f"{value:.2f}")
        print("\nGROUP ROBUSTNESS")
        print(display.to_string(index=False))

    if sensitivity_results.empty:
        print("\nNo sensitivity results")
    else:
        display = sensitivity_results.copy()
        display["win_rate"] = display["win_rate"].map(lambda value: f"{value:.1%}")
        display["roi"] = display["roi"].map(lambda value: f"{value:.1%}")
        display["avg_edge"] = display["avg_edge"].map(lambda value: f"{value:.1%}")
        display["avg_profit"] = display["avg_profit"].map(lambda value: f"{value:.2f}")
        print("\nNEARBY THRESHOLD SENSITIVITY")
        print(display.to_string(index=False))


def print_calibration_summary(calibration_results: pd.DataFrame):
    """Print probability calibration headline metrics."""
    print("\nCALIBRATION SUMMARY")
    if calibration_results.empty:
        print("No calibration data")
        return

    non_empty = calibration_results[calibration_results["count"] > 0].copy()
    if non_empty.empty:
        print("No non-empty probability buckets")
        return

    mean_abs_error = non_empty["calibration_error"].mean()
    worst_bucket = non_empty.sort_values("calibration_error", ascending=False).iloc[0]
    print(f"Mean absolute calibration error: {mean_abs_error:.1%}")
    print(
        "Worst bucket: "
        f"{worst_bucket['probability_bucket']} "
        f"(count={int(worst_bucket['count'])}, "
        f"avg_pred={worst_bucket['avg_predicted_prob']:.1%}, "
        f"actual={worst_bucket['actual_over_4_5_rate']:.1%}, "
        f"error={worst_bucket['calibration_error']:.1%})"
    )


def print_calibration_comparison(calibration_comparison: pd.DataFrame):
    """Print raw versus calibrated probability and ROI comparison."""
    print("\nCALIBRATION COMPARISON")
    if calibration_comparison.empty:
        print("No calibration comparison data")
        return

    display = calibration_comparison.copy()
    for column in [
        "mean_abs_calibration_error",
        "worst_bucket_error",
        "roi",
        "win_rate",
        "avg_edge",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.1%}")
    print(display.to_string(index=False))


def _format_diagnostic_for_print(diagnostic: pd.DataFrame) -> pd.DataFrame:
    display = diagnostic.copy()
    for column in ["win_rate", "roi", "avg_edge"]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.1%}")
    if "avg_profit" in display.columns:
        display["avg_profit"] = display["avg_profit"].map(lambda value: f"{value:.2f}")
    return display


def print_best_worst_diagnostics(diagnostics: dict[str, pd.DataFrame]):
    """Print best and worst groups by ROI for each diagnostic dimension."""
    print("\nDIAGNOSTIC BEST/WORST GROUPS")
    for name, diagnostic in diagnostics.items():
        if diagnostic.empty:
            print(f"\n{name}: no data")
            continue

        best = diagnostic.sort_values(
            ["roi", "total_bets"], ascending=[False, False]
        ).head(3)
        worst = diagnostic.sort_values(
            ["roi", "total_bets"], ascending=[True, False]
        ).head(3)

        print(f"\n{name.upper()} - BEST")
        print(_format_diagnostic_for_print(best).to_string(index=False))
        print(f"\n{name.upper()} - WORST")
        print(_format_diagnostic_for_print(worst).to_string(index=False))


def main():
    (
        results,
        realistic_summary,
        fixed_summary,
        audit,
        threshold_results,
        diagnostics,
        strategy_results,
        logistic_strategy_results,
        ultra_optimization_results,
        robustness_results,
        sensitivity_results,
        calibration_results,
        calibration_comparison,
        robustness_verdict,
    ) = run_backtest()
    print_summary(realistic_summary, fixed_summary, audit)
    print_threshold_table(threshold_results)
    print_strategy_table(strategy_results)
    print_logistic_strategy_table(logistic_strategy_results)
    print_ultra_value_optimization(ultra_optimization_results)
    print_best_strategy_robustness(
        robustness_results, sensitivity_results, robustness_verdict
    )
    print_calibration_summary(calibration_results)
    print_calibration_comparison(calibration_comparison)
    print_best_worst_diagnostics(diagnostics)

    if not results.empty:
        print("\nSAMPLE BETS")
        print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
