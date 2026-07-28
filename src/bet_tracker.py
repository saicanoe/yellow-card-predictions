from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_PATH = PROJECT_ROOT / "data" / "bet_tracking.csv"
TRACKING_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "Pick",
    "Line",
    "Odds",
    "Stake",
    "Edge",
    "Confidence",
    "FinalCards",
    "Result",
    "Profit",
]


def initialize_bet_tracking() -> pd.DataFrame:
    """Create the bet tracking CSV if needed and return its contents."""
    if not TRACKING_PATH.exists():
        TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=TRACKING_COLUMNS).to_csv(TRACKING_PATH, index=False)

    tracking = pd.read_csv(TRACKING_PATH)
    for column in TRACKING_COLUMNS:
        if column not in tracking.columns:
            tracking[column] = pd.NA

    tracking["Pick"] = tracking.apply(_normalize_pick_label, axis=1)
    tracking = tracking[TRACKING_COLUMNS]
    tracking.to_csv(TRACKING_PATH, index=False)
    return tracking


def _pick_from_signal(signal: str) -> str:
    signal = str(signal).upper()
    if "UNDER" in signal:
        return "UNDER"
    if "OVER" in signal:
        return "OVER"
    return ""


def _format_line(line) -> str:
    line = pd.to_numeric(line, errors="coerce")
    if pd.isna(line):
        return "4.5"
    if float(line).is_integer():
        return str(int(line))
    return str(float(line)).rstrip("0").rstrip(".")


def _normalize_pick_label(row: pd.Series) -> str:
    pick = str(row.get("Pick", "")).strip().upper()
    if not pick:
        return pick
    if " " in pick:
        return pick
    if pick in {"UNDER", "OVER"}:
        return f"{pick} {_format_line(row.get('Line', 4.5))}"
    return pick


def _date_key(value) -> str:
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _profit_for_result(result, odds, stake) -> float:
    result = str(result).strip().upper()
    odds = pd.to_numeric(odds, errors="coerce")
    stake = pd.to_numeric(stake, errors="coerce")

    if pd.isna(odds) or pd.isna(stake):
        return 0.0
    if result == "WIN":
        return float(stake * (odds - 1))
    if result == "LOSS":
        return float(-stake)
    if result == "PUSH":
        return 0.0
    return 0.0


def _result_from_final_cards(row: pd.Series) -> str:
    final_cards = pd.to_numeric(row.get("FinalCards"), errors="coerce")
    if pd.isna(final_cards):
        return str(row.get("Result", "PENDING")).strip().upper() or "PENDING"

    line = pd.to_numeric(row.get("Line"), errors="coerce")
    if pd.isna(line):
        line = 4.5

    pick = str(row.get("Pick", "")).strip().upper()
    if pick.startswith("UNDER"):
        if final_cards < line:
            return "WIN"
        if final_cards > line:
            return "LOSS"
        return "PUSH"

    if pick.startswith("OVER"):
        if final_cards > line:
            return "WIN"
        if final_cards < line:
            return "LOSS"
        return "PUSH"

    return "PENDING"


def log_top_bets(top_bets: pd.DataFrame) -> pd.DataFrame:
    """Append new top bets while avoiding duplicate match/pick rows."""
    tracking = initialize_bet_tracking()
    if top_bets.empty:
        return tracking

    rows = []
    for _, bet in top_bets.iterrows():
        pick = _pick_from_signal(bet.get("signal", ""))
        if not pick:
            continue

        line = bet.get("Line", bet.get("line", 4.5))
        odds_column = "UnderOdds" if pick == "UNDER" else "OverOdds"
        rows.append(
            {
                "Date": bet.get("Date"),
                "HomeTeam": bet.get("HomeTeam"),
                "AwayTeam": bet.get("AwayTeam"),
                "Pick": f"{pick} {_format_line(line)}",
                "Line": line,
                "Odds": bet.get(odds_column, 0),
                "Stake": 1.0,
                "Edge": bet.get("value_edge", bet.get("edge", pd.NA)),
                "Confidence": bet.get("confidence", pd.NA),
                "FinalCards": pd.NA,
                "Result": "PENDING",
                "Profit": 0.0,
            }
        )

    if not rows:
        return tracking

    new_bets = pd.DataFrame(rows, columns=TRACKING_COLUMNS)
    tracking["_dedupe_date"] = tracking["Date"].apply(_date_key)
    new_bets["_dedupe_date"] = new_bets["Date"].apply(_date_key)

    rows_to_append = []
    for _, new_bet in new_bets.iterrows():
        duplicate = (
            (tracking["_dedupe_date"] == new_bet["_dedupe_date"])
            & (tracking["HomeTeam"] == new_bet["HomeTeam"])
            & (tracking["AwayTeam"] == new_bet["AwayTeam"])
            & (tracking["Pick"] == new_bet["Pick"])
        )

        if duplicate.any():
            duplicate_index = tracking[duplicate].index[0]
            for column in ["Line", "Odds", "Edge", "Confidence"]:
                if pd.isna(tracking.at[duplicate_index, column]):
                    tracking.at[duplicate_index, column] = new_bet[column]
        else:
            rows_to_append.append(new_bet.drop(labels=["_dedupe_date"]))

    append_frame = pd.DataFrame(rows_to_append, columns=TRACKING_COLUMNS)
    combined = pd.concat(
        [tracking.drop(columns=["_dedupe_date"]), append_frame], ignore_index=True
    )
    combined["_dedupe_date"] = pd.to_datetime(
        combined["Date"], dayfirst=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    combined = combined.drop_duplicates(
        subset=["_dedupe_date", "HomeTeam", "AwayTeam", "Pick"],
        keep="first",
    )
    combined = combined.drop(columns=["_dedupe_date"])
    combined = combined[TRACKING_COLUMNS]
    combined.to_csv(TRACKING_PATH, index=False)
    return combined


def evaluate_tracking(tracking: pd.DataFrame) -> pd.DataFrame:
    """Return tracking rows with automatic Result and Profit recalculated."""
    tracking = tracking.copy()
    for column in TRACKING_COLUMNS:
        if column not in tracking.columns:
            tracking[column] = pd.NA

    tracking = tracking[TRACKING_COLUMNS]
    tracking["Pick"] = tracking.apply(_normalize_pick_label, axis=1)
    tracking["Stake"] = pd.to_numeric(tracking["Stake"], errors="coerce").fillna(1.0)
    tracking["Odds"] = pd.to_numeric(tracking["Odds"], errors="coerce").fillna(0.0)
    tracking["Line"] = pd.to_numeric(tracking["Line"], errors="coerce").fillna(4.5)
    tracking["FinalCards"] = pd.to_numeric(tracking["FinalCards"], errors="coerce")
    tracking["Edge"] = pd.to_numeric(tracking["Edge"], errors="coerce")
    tracking["Confidence"] = tracking["Confidence"].fillna("UNKNOWN")
    tracking["Result"] = tracking.apply(_result_from_final_cards, axis=1)
    tracking["Profit"] = tracking.apply(
        lambda row: _profit_for_result(row["Result"], row["Odds"], row["Stake"]),
        axis=1,
    )
    return tracking[TRACKING_COLUMNS]


def update_results() -> pd.DataFrame:
    """Recalculate result and profit from final card totals."""
    tracking = evaluate_tracking(initialize_bet_tracking())
    tracking.to_csv(TRACKING_PATH, index=False)
    return tracking


def calculate_summary_for_tracking(tracking: pd.DataFrame) -> dict:
    """Calculate bet count, result counts, profit, edge, and ROI."""
    tracking = evaluate_tracking(tracking)
    if tracking.empty:
        return {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "total_profit": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "avg_edge": 0.0,
            "roi_by_confidence": {},
        }

    results = tracking["Result"].fillna("PENDING").str.upper().str.strip()
    settled_mask = results.isin(["WIN", "LOSS", "PUSH"])
    settled = tracking[settled_mask]
    total_staked = pd.to_numeric(settled["Stake"], errors="coerce").fillna(0).sum()
    total_profit = pd.to_numeric(tracking["Profit"], errors="coerce").fillna(0).sum()
    wins = int((results == "WIN").sum())
    losses = int((results == "LOSS").sum())
    settled_decisions = wins + losses
    edge = pd.to_numeric(tracking["Edge"], errors="coerce")

    roi_by_confidence = {}
    for confidence, group in settled.groupby(
        settled["Confidence"].fillna("UNKNOWN"), dropna=False
    ):
        stake = pd.to_numeric(group["Stake"], errors="coerce").fillna(0).sum()
        profit = pd.to_numeric(group["Profit"], errors="coerce").fillna(0).sum()
        roi_by_confidence[str(confidence)] = float(profit / stake) if stake else 0.0

    return {
        "total_bets": int(len(tracking)),
        "wins": wins,
        "losses": losses,
        "pending": int((results == "PENDING").sum()),
        "total_profit": float(total_profit),
        "roi": float(total_profit / total_staked) if total_staked else 0.0,
        "win_rate": float(wins / settled_decisions) if settled_decisions else 0.0,
        "avg_edge": float(edge.mean()) if edge.notna().any() else 0.0,
        "roi_by_confidence": roi_by_confidence,
    }


def calculate_summary() -> dict:
    """Calculate bet count, result counts, profit, edge, and ROI from CSV."""
    return calculate_summary_for_tracking(update_results())


def save_tracking_edits(tracking: pd.DataFrame) -> pd.DataFrame:
    """Persist dashboard edits and refresh calculated profits."""
    tracking = tracking.copy()
    for column in TRACKING_COLUMNS:
        if column not in tracking.columns:
            tracking[column] = pd.NA

    tracking = evaluate_tracking(tracking[TRACKING_COLUMNS])
    tracking.to_csv(TRACKING_PATH, index=False)
    return tracking
