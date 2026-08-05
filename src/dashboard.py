import pandas as pd
import streamlit as st
from datetime import date

from live_predictor import (
    LivePredictionError,
    generate_live_prediction,
)
from live_betting_value import (
    MANUAL_TEST_ODDS,
    UNAVAILABLE_ODDS,
    VERIFIED_CARD_MARKET,
    calculate_live_betting_value,
)

from api_football import (
    APIFootballError,
    get_fixture_label,
    get_fixture_lineup_data,
    get_normalized_fixtures_by_date,
)

from bet_tracker import (
    TRACKING_COLUMNS,
    calculate_summary,
    calculate_summary_for_tracking,
    evaluate_tracking,
    initialize_bet_tracking,
    save_live_prediction,
    save_tracking_edits,
)
from player_risk import (
    OUTPUT_PATH as PLAYER_RISK_PATH,
    generate_live_player_card_risks,
)
from player_stats_api import PlayerStatsAPIError

TEST_LIVE_FIXTURE = {
    "fixture_id": -1,
    "date": "2026-08-03T15:00:00-04:00",
    "status": "NS",
    "referee": "J Brooks",
    "league_id": 39,
    "league": "Premier League",
    "country": "England",
    "season": 2026,
    "home_team_id": 42,
    "home_team": "Arsenal",
    "away_team_id": 49,
    "away_team": "Chelsea",
}

TEST_LIVE_LINEUPS = [
    {"team_id": 42, "team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player_id": 900001, "player": "David Raya", "number": 1, "position": "G", "grid": "1:1"},
    {"team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player": "Jurrien Timber", "number": 12, "position": "D", "grid": "2:1"},
    {"team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player": "William Saliba", "number": 2, "position": "D", "grid": "2:2"},
    {"team_id": 42, "team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player_id": 900002, "player": "Gabriel Magalhaes", "number": 6, "position": "D", "grid": "2:3"},
    {"team_id": 42, "team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player_id": 900003, "player": "Myles Lewis-Skelly", "number": 49, "position": "D", "grid": "2:4"},
    {"team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player": "Declan Rice", "number": 41, "position": "M", "grid": "3:1"},
    {"team_id": 42, "team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player_id": 900004, "player": "Martin Odegaard", "number": 8, "position": "M", "grid": "3:2"},
    {"team_id": 42, "team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player_id": 900005, "player": "Mikel Merino", "number": 23, "position": "M", "grid": "3:3"},
    {"team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player": "Bukayo Saka", "number": 7, "position": "F", "grid": "4:1"},
    {"team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player": "Kai Havertz", "number": 29, "position": "F", "grid": "4:2"},
    {"team": "Arsenal", "formation": "4-3-3", "lineup_type": "Starter", "player": "Gabriel Martinelli", "number": 11, "position": "F", "grid": "4:3"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Robert Sanchez", "number": 1, "position": "G", "grid": "1:1"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Malo Gusto", "number": 27, "position": "D", "grid": "2:1"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Wesley Fofana", "number": 29, "position": "D", "grid": "2:2"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Levi Colwill", "number": 6, "position": "D", "grid": "2:3"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Marc Cucurella", "number": 3, "position": "D", "grid": "2:4"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Moisés Caicedo", "number": 25, "position": "M", "grid": "3:1"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Enzo Fernandez", "number": 8, "position": "M", "grid": "3:2"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Pedro Neto", "number": 7, "position": "M", "grid": "4:1"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Cole Palmer", "number": 20, "position": "M", "grid": "4:2"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Jadon Sancho", "number": 19, "position": "M", "grid": "4:3"},
    {"team": "Chelsea", "formation": "4-2-3-1", "lineup_type": "Starter", "player": "Nicolas Jackson", "number": 15, "position": "F", "grid": "5:1"},
]

st.set_page_config(
    page_title="Yellow Card Predictions",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Condensed:wght@600;700&display=swap');

    :root {
        --yc-bg: #020617;
        --yc-bg-glow-a: rgba(14, 165, 233, 0.16);
        --yc-bg-glow-b: rgba(51, 65, 85, 0.32);
        --yc-surface: rgba(15, 23, 42, 0.94);
        --yc-surface-soft: rgba(15, 23, 42, 0.72);
        --yc-surface-deep: rgba(2, 6, 23, 0.72);
        --yc-border: rgba(148, 163, 184, 0.16);
        --yc-border-strong: rgba(148, 163, 184, 0.28);
        --yc-text: #e2e8f0;
        --yc-muted: #94a3b8;
        --yc-heading: #f8fafc;
        --yc-accent: #38bdf8;
        --yc-accent-strong: #0ea5e9;
        --yc-accent-deep: #0369a1;
        --yc-good-bg: rgba(34, 197, 94, 0.14);
        --yc-good-fg: #86efac;
        --yc-good-bd: rgba(34, 197, 94, 0.35);
        --yc-warn-bg: rgba(245, 158, 11, 0.14);
        --yc-warn-fg: #fcd34d;
        --yc-warn-bd: rgba(245, 158, 11, 0.35);
        --yc-bad-bg: rgba(239, 68, 68, 0.14);
        --yc-bad-fg: #fca5a5;
        --yc-bad-bd: rgba(239, 68, 68, 0.35);
        --yc-neutral-bg: rgba(56, 189, 248, 0.14);
        --yc-neutral-fg: #7dd3fc;
        --yc-neutral-bd: rgba(56, 189, 248, 0.35);
        --yc-radius: 16px;
        --yc-radius-sm: 12px;
        --yc-shadow: 0 12px 36px rgba(0, 0, 0, 0.28);
        --yc-font: 'IBM Plex Sans', sans-serif;
        --yc-font-display: 'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif;
        --yc-space-xs: 0.35rem;
        --yc-space-sm: 0.75rem;
        --yc-space-md: 1.25rem;
        --yc-space-lg: 1.75rem;
    }

    html, body, [class*="css"] {
        font-family: var(--yc-font);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, var(--yc-bg-glow-a), transparent 38%),
            radial-gradient(circle at top right, var(--yc-bg-glow-b), transparent 32%),
            var(--yc-bg);
        color: var(--yc-text);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
        max-width: 1440px;
    }

    h1 {
        font-family: var(--yc-font-display) !important;
        font-size: clamp(1.85rem, 4vw, 2.55rem) !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em;
        color: var(--yc-heading);
        margin-bottom: 0.35rem !important;
        line-height: 1.1 !important;
    }

    h2, h3, h4 {
        font-family: var(--yc-font-display) !important;
        color: var(--yc-heading);
        letter-spacing: -0.02em;
    }

    .subtitle {
        color: var(--yc-muted);
        font-size: 1rem;
        line-height: 1.55;
        margin-bottom: var(--yc-space-md);
        max-width: 52rem;
    }

    .section-header {
        margin: var(--yc-space-md) 0 var(--yc-space-sm);
        padding-bottom: 0.55rem;
        border-bottom: 1px solid var(--yc-border);
    }

    .section-header.featured {
        border-bottom-color: rgba(56, 189, 248, 0.35);
    }

    .section-kicker {
        color: var(--yc-accent);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .section-title {
        font-family: var(--yc-font-display);
        color: var(--yc-heading);
        font-size: clamp(1.2rem, 2.4vw, 1.55rem);
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .section-subtitle {
        color: var(--yc-muted);
        font-size: 0.92rem;
        margin-top: 0.3rem;
        line-height: 1.45;
        max-width: 48rem;
    }

    .panel {
        background: linear-gradient(180deg, var(--yc-surface), var(--yc-surface-soft));
        border: 1px solid var(--yc-border);
        border-radius: var(--yc-radius);
        padding: 1.15rem 1.25rem 1.05rem;
        margin-bottom: var(--yc-space-md);
        box-shadow: var(--yc-shadow);
    }

    .panel-accent {
        border-color: rgba(56, 189, 248, 0.28);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(56, 189, 248, 0.08);
    }

    .panel-value {
        border-color: rgba(34, 197, 94, 0.22);
    }

    .metric-card {
        background: linear-gradient(180deg, var(--yc-surface), rgba(2, 6, 23, 0.95));
        border: 1px solid var(--yc-border);
        border-radius: var(--yc-radius-sm);
        padding: 14px 16px;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        margin-bottom: var(--yc-space-xs);
        height: 100%;
    }

    .match-card {
        background: linear-gradient(180deg, var(--yc-surface), var(--yc-surface-soft));
        border: 1px solid var(--yc-border);
        border-radius: 18px;
        padding: 1.15rem 1.25rem 0.85rem;
        margin-bottom: 0.85rem;
        box-shadow: var(--yc-shadow);
    }

    .match-card-top {
        display: grid;
        grid-template-columns: minmax(0, 1.7fr) minmax(0, 1fr);
        gap: 1rem 1.25rem;
        align-items: start;
    }

    .match-card-title {
        font-family: var(--yc-font-display);
        color: var(--yc-heading);
        font-size: clamp(1.15rem, 2.2vw, 1.35rem);
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 4px 0 10px;
        line-height: 1.2;
    }

    .match-card-meta {
        margin-top: 0.55rem;
    }

    .match-ref-name {
        font-size: 1.02rem;
        font-weight: 600;
        color: var(--yc-heading);
        margin-top: 4px;
        line-height: 1.35;
    }

    .match-metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin-top: 0.95rem;
        padding-top: 0.85rem;
        border-top: 1px solid var(--yc-border);
    }

    .match-metric {
        background: rgba(2, 6, 23, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 12px 14px;
    }

    .featured-prediction {
        margin-top: 0.35rem;
    }

    .featured-metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.85rem;
    }

    .featured-metric {
        background: rgba(2, 6, 23, 0.45);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 12px;
        padding: 14px 16px;
    }

    .featured-metric .big-number {
        font-size: clamp(1.45rem, 3vw, 1.95rem);
    }

    .recommend-panel {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(6, 78, 59, 0.22));
        border: 1px solid rgba(34, 197, 94, 0.35);
        border-radius: var(--yc-radius);
        padding: 1.15rem 1.25rem 1.05rem;
        margin: 0.75rem 0 1rem;
        box-shadow: var(--yc-shadow);
    }

    .recommend-title {
        font-family: var(--yc-font-display);
        color: var(--yc-heading);
        font-size: clamp(1.2rem, 2.4vw, 1.55rem);
        font-weight: 700;
        margin: 4px 0 0.85rem;
    }

    .side-compare-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.5rem 0 0.85rem;
    }

    .side-card {
        background: var(--yc-surface-soft);
        border: 1px solid var(--yc-border);
        border-radius: 14px;
        padding: 14px 16px;
    }

    .side-card.recommended {
        border-color: rgba(34, 197, 94, 0.4);
        box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.12);
    }

    .side-card-title {
        font-family: var(--yc-font-display);
        font-weight: 700;
        color: var(--yc-heading);
        margin-bottom: 0.65rem;
    }

    .side-stat-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.28rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    .side-stat-row:last-child {
        border-bottom: none;
    }

    .side-stat-label {
        color: var(--yc-muted);
    }

    .side-stat-value {
        font-weight: 600;
        color: var(--yc-heading);
        text-align: right;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 4px;
        letter-spacing: 0.02em;
        vertical-align: middle;
    }

    .good { background: var(--yc-good-bg); color: var(--yc-good-fg); border: 1px solid var(--yc-good-bd); }
    .warn { background: var(--yc-warn-bg); color: var(--yc-warn-fg); border: 1px solid var(--yc-warn-bd); }
    .bad { background: var(--yc-bad-bg); color: var(--yc-bad-fg); border: 1px solid var(--yc-bad-bd); }
    .neutral { background: var(--yc-neutral-bg); color: var(--yc-neutral-fg); border: 1px solid var(--yc-neutral-bd); }

    .small-label {
        color: var(--yc-muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 600;
    }

    .big-number {
        color: var(--yc-heading);
        font-family: var(--yc-font-display);
        font-size: clamp(1.35rem, 2.5vw, 1.75rem);
        font-weight: 700;
        margin-top: 2px;
        line-height: 1.15;
    }

    .reason-box {
        background: var(--yc-surface-deep);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 14px 16px;
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-top: 0.75rem;
    }

    .overview-line {
        color: var(--yc-muted);
        font-size: 0.9rem;
        margin: 0.15rem 0 0.65rem;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--yc-border);
    }

    div[data-testid="stMetric"] {
        background: var(--yc-surface-soft);
        border: 1px solid var(--yc-border);
        border-radius: var(--yc-radius-sm);
        padding: 12px 14px 10px;
    }

    div[data-testid="stMetric"] label {
        color: var(--yc-muted) !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--yc-heading) !important;
        font-family: var(--yc-font-display);
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 0.5rem;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--yc-surface);
        border-radius: var(--yc-radius-sm);
        padding: 10px 16px;
        border: 1px solid var(--yc-border);
        color: #cbd5e1;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--yc-accent-strong), var(--yc-accent-deep));
        color: white;
        border: 1px solid transparent;
    }

    div[data-testid="stRadio"] > div {
        gap: 0.35rem 0.85rem;
        flex-wrap: wrap;
    }

    div[data-testid="stRadio"] label {
        background: var(--yc-surface-soft);
        border: 1px solid var(--yc-border);
        border-radius: 999px;
        padding: 0.35rem 0.75rem !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1px solid var(--yc-border-strong) !important;
        min-height: 2.6rem;
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--yc-accent-strong), var(--yc-accent-deep)) !important;
        border: 1px solid transparent !important;
        color: #fff !important;
    }

    div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input {
        background-color: rgba(2, 6, 23, 0.55) !important;
        border-color: var(--yc-border) !important;
        border-radius: 10px !important;
        color: var(--yc-text) !important;
    }

    .stExpander {
        border: 1px solid var(--yc-border) !important;
        border-radius: var(--yc-radius-sm) !important;
        background: var(--yc-surface-soft);
    }

    @media (max-width: 900px) {
        .match-metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .featured-metric-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.85rem;
            padding-right: 0.85rem;
            padding-top: 1rem;
        }

        .match-card,
        .panel,
        .recommend-panel {
            padding: 0.95rem 1rem;
            border-radius: 14px;
        }

        .match-card-top,
        .side-compare-grid {
            grid-template-columns: 1fr;
        }

        .big-number {
            font-size: 1.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 0.9rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_data():
    predictions = pd.read_csv("data/predictions.csv")
    top_bets = pd.read_csv("data/top_bets.csv")
    try:
        ultra_top_bets = pd.read_csv("data/ultra_top_bets.csv")
    except FileNotFoundError:
        ultra_top_bets = pd.DataFrame()

    predictions["Date"] = pd.to_datetime(
        predictions["Date"], dayfirst=True, errors="coerce"
    )
    top_bets["Date"] = pd.to_datetime(top_bets["Date"], dayfirst=True, errors="coerce")
    if not ultra_top_bets.empty:
        ultra_top_bets["Date"] = pd.to_datetime(
            ultra_top_bets["Date"], dayfirst=True, errors="coerce"
        )

    today = pd.Timestamp(date.today())
    predictions = predictions[predictions["Date"] >= today].copy()
    top_bets = top_bets[top_bets["Date"] >= today].copy()
    if not ultra_top_bets.empty:
        ultra_top_bets = ultra_top_bets[ultra_top_bets["Date"] >= today].copy()

    predictions = predictions.sort_values(["Date", "HomeTeam"])
    top_bets = top_bets.sort_values("value_edge", ascending=False)
    if not ultra_top_bets.empty:
        ultra_top_bets = ultra_top_bets.sort_values("value_edge", ascending=False)

    return predictions, top_bets, ultra_top_bets


def load_tracking_data():
    tracking = initialize_bet_tracking()
    if not tracking.empty:
        tracking["Date"] = pd.to_datetime(
            tracking["Date"], dayfirst=True, errors="coerce"
        )
    return tracking


def load_player_risks():
    if not PLAYER_RISK_PATH.exists():
        return pd.DataFrame()

    risks = pd.read_csv(PLAYER_RISK_PATH)
    if not risks.empty:
        risks["Date"] = pd.to_datetime(risks["Date"], dayfirst=True, errors="coerce")
        risks = risks.sort_values(
            ["Date", "HomeTeam", "AwayTeam", "RiskScore"],
            ascending=[True, True, True, False],
        )
    return risks


def is_missing_referee(value) -> bool:
    return pd.isna(value) or str(value).strip().lower() in ["", "nan", "none"]


def referee_display(value) -> str:
    return "Missing referee data" if is_missing_referee(value) else str(value).strip()


def badge(text, kind="neutral"):
    return f"<span class='badge {kind}'>{text}</span>"


def data_source_badges(row):
    odds_source = str(row.get("OddsSource", "DEFAULT ODDS")).strip().upper()
    if odds_source == "API ODDS":
        odds_badge = badge("API ODDS", "good")
    elif odds_source == "MANUAL ODDS":
        odds_badge = badge("MANUAL ODDS", "neutral")
    else:
        odds_badge = badge("DEFAULT ODDS", "warn")

    ref_badge = (
        badge("NO REF DATA", "bad")
        if is_missing_referee(row.get("Referee"))
        else badge("REF DATA", "good")
    )
    lineup_badge = (
        badge("LINEUPS LOADED", "good")
        if str(row.get("LineupsLoaded", "")).strip().upper() == "YES"
        else ""
    )
    return f"{odds_badge} {ref_badge} {lineup_badge}"


def fmt_pct(value):
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "N/A"


def fmt_num(value):
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"


def section_header(title, subtitle=None, kicker=None, featured=False):
    featured_class = " featured" if featured else ""
    kicker_html = (
        f'<div class="section-kicker">{kicker}</div>' if kicker else ""
    )
    subtitle_html = (
        f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="section-header{featured_class}">
            {kicker_html}
            <div class="section-title">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def match_metric_html(label, value, css_class="match-metric"):
    return (
        f"<div class='{css_class}'>"
        f"<div class='small-label'>{label}</div>"
        f"<div class='big-number'>{value}</div>"
        f"</div>"
    )


def side_stat_html(label, value):
    return (
        f"<div class='side-stat-row'>"
        f"<span class='side-stat-label'>{label}</span>"
        f"<span class='side-stat-value'>{value}</span>"
        f"</div>"
    )


def match_card(row):
    confidence = str(row.get("confidence", ""))
    signal = str(row.get("signal", ""))
    value_bet = str(row.get("value_bet", ""))

    if confidence == "HIGH CONFIDENCE":
        confidence_kind = "good"
    elif "no ref data" in confidence.lower():
        confidence_kind = "bad"
    else:
        confidence_kind = "warn"
    signal_kind = "good" if "STRONG" in signal else "neutral"
    value_kind = "good" if value_bet == "YES" else "bad"

    match_label = f"{row['HomeTeam']} vs {row['AwayTeam']}"
    date_text = row["Date"].strftime("%m/%d/%Y")
    league_text = row.get("Div", "") or ""

    st.markdown(
        f"""
        <div class="match-card">
            <div class="match-card-top">
                <div>
                    <div class="small-label">{league_text} &middot; {date_text}</div>
                    <div class="match-card-title">{match_label}</div>
                    <div>
                        {badge(signal, signal_kind)}
                        {badge(confidence, confidence_kind)}
                        {badge('VALUE: ' + value_bet, value_kind)}
                    </div>
                </div>
                <div>
                    <div class="small-label">Referee</div>
                    <div class="match-ref-name">{referee_display(row.get('Referee'))}</div>
                    <div class="match-card-meta">{data_source_badges(row)}</div>
                </div>
            </div>
            <div class="match-metric-grid">
                {match_metric_html('Predicted Cards', fmt_num(row['predicted_cards']))}
                {match_metric_html('Under 4.5 Model Prob', fmt_pct(row.get('under_model_prob')))}
                {match_metric_html('Book Prob', fmt_pct(row.get('under_book_prob')))}
                {match_metric_html('Value Edge', fmt_pct(row.get('value_edge')))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Reasoning & details · {match_label}", expanded=False):
        st.markdown(
            f"""
            <div class="reason-box">
                <b>Reasoning</b><br>
                Ref average cards: {fmt_num(row.get("avg_total_cards"))}<br>
                Ref over 4.5 rate: {fmt_pct(row.get("over_4_5_rate"))}<br>
                Model over 4.5 probability: {fmt_pct(row.get("over_4_5_prob"))}<br>
                Edge vs 4.5 line: {fmt_num(row.get("edge"))}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_live_prediction_summary(prediction):
    st.markdown(
        f"""
        <div class="panel panel-accent featured-prediction">
            <div class="small-label">Live model output</div>
            <div class="featured-metric-grid">
                {match_metric_html('Predicted Cards', fmt_num(prediction['predicted_cards']), 'featured-metric')}
                {match_metric_html('Over 4.5 Probability', fmt_pct(prediction['over_4_5_probability']), 'featured-metric')}
                {match_metric_html('Under 4.5 Probability', fmt_pct(prediction['under_4_5_probability']), 'featured-metric')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile_found = prediction["referee_profile_found"]
    profile_badge = (
        badge("REFEREE PROFILE FOUND", "good")
        if profile_found
        else badge("LEAGUE AVERAGE PROFILE", "warn")
    )
    referee_name = prediction.get("referee_model") or "No referee assigned"

    with st.expander("Referee profile & team mapping", expanded=False):
        st.markdown(
            f"""
            <div class="reason-box">
                <div style="margin-bottom: 10px;">{profile_badge}</div>
                <b>Referee profile status:</b> {referee_name}<br>
                <b>Historical team mapping:</b><br>
                {prediction["home_team_api"]} &rarr; {prediction["home_team_model"]}<br>
                {prediction["away_team_api"]} &rarr; {prediction["away_team_model"]}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_betting_value_sides(value, stake):
    sides = value.get("sides") or {}
    recommendation = value.get("recommendation")

    if recommendation and recommendation in sides:
        metrics = sides[recommendation]
        st.markdown(
            f"""
            <div class="recommend-panel">
                <div class="small-label">Recommended side</div>
                <div class="recommend-title">{recommendation}</div>
                <div class="featured-metric-grid">
                    {match_metric_html('Model Prob', fmt_pct(metrics['model_probability']), 'featured-metric')}
                    {match_metric_html('Edge', fmt_pct(metrics['probability_edge']), 'featured-metric')}
                    {match_metric_html('Expected Value', fmt_pct(metrics['expected_value_per_unit']), 'featured-metric')}
                </div>
                <div class="reason-box">
                    Decimal odds: {metrics['decimal_odds']:.2f}<br>
                    No-vig market probability: {fmt_pct(metrics['no_vig_implied_probability'])}<br>
                    Expected profit ({stake:g} stake): {metrics['expected_profit']:+.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No positive EV")

    side_cards = []
    for side_name, metrics in sides.items():
        recommended_class = " recommended" if side_name == recommendation else ""
        side_cards.append(
            f"""
            <div class="side-card{recommended_class}">
                <div class="side-card-title">{side_name}</div>
                {side_stat_html('Model Probability', fmt_pct(metrics['model_probability']))}
                {side_stat_html('Decimal Odds', f"{metrics['decimal_odds']:.2f}")}
                {side_stat_html('No-Vig Market Prob', fmt_pct(metrics['no_vig_implied_probability']))}
                {side_stat_html('Edge', fmt_pct(metrics['probability_edge']))}
                {side_stat_html('Expected Value', fmt_pct(metrics['expected_value_per_unit']))}
                {side_stat_html(f'Expected Profit ({stake:g})', f"{metrics['expected_profit']:+.2f}")}
            </div>
            """
        )

    if side_cards:
        st.markdown(
            f'<div class="side-compare-grid">{"".join(side_cards)}</div>',
            unsafe_allow_html=True,
        )

    rows = []
    for side, metrics in sides.items():
        rows.append(
            {
                "Side": side,
                "Model Probability": fmt_pct(metrics["model_probability"]),
                "Decimal Odds": f"{metrics['decimal_odds']:.2f}",
                "No-Vig Market Probability": fmt_pct(
                    metrics["no_vig_implied_probability"]
                ),
                "Edge": fmt_pct(metrics["probability_edge"]),
                "Expected Value": fmt_pct(metrics["expected_value_per_unit"]),
                f"Expected Profit ({stake:g} stake)": (
                    f"{metrics['expected_profit']:+.2f}"
                ),
            }
        )
    if rows:
        with st.expander("Full value table", expanded=False):
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_match_list(rows, empty_message):
    if rows.empty:
        st.info(empty_message)
        return

    for _, row in rows.iterrows():
        match_card(row)


def render_predictions_hub(predictions, top_bets, ultra_top_bets, high_conf, missing_ref):
    section_header(
        "Predictions",
        "Browse upcoming matches by signal quality. Filters only change what is shown.",
        kicker="Batch signals",
    )

    view = st.radio(
        "Prediction view",
        [
            "Top Bets",
            "Ultra Value",
            "With Ref Data",
            "Without Ref Data",
            "All Predictions",
        ],
        horizontal=True,
        key="predictions_view",
    )

    if view == "Top Bets":
        st.subheader("Top Bets")
        render_match_list(
            top_bets,
            "No top bets available for today or future matches.",
        )
    elif view == "Ultra Value":
        render_ultra_value(ultra_top_bets)
    elif view == "With Ref Data":
        st.subheader("Matches With Referee Data")
        render_match_list(high_conf, "No high-confidence matches available.")
    elif view == "Without Ref Data":
        st.subheader("Matches Without Referee Data")
        st.caption(
            "These matches are shown separately because the model is using fallback average referee values."
        )
        if missing_ref.empty:
            st.success("No matches are missing referee data.")
        else:
            render_match_list(missing_ref, "No matches are missing referee data.")
    else:
        st.subheader("All Upcoming Predictions")
        st.dataframe(predictions, width="stretch")


def render_bet_tracker():
    summary = calculate_summary()
    tracking = load_tracking_data()

    section_header(
        "Prediction History",
        "Saved live picks and outcomes. Enter Final Cards after a match finishes — "
        "Result and Profit are calculated from Pick, Line, Odds, and Stake.",
        kicker="Tracker",
        featured=True,
    )

    row_a = st.columns(4, gap="small")
    row_a[0].metric("Total Saved", summary["total_bets"])
    row_a[1].metric("Wins", summary["wins"])
    row_a[2].metric("Losses", summary["losses"])
    row_a[3].metric("Pending", summary["pending"])

    row_b = st.columns(4, gap="small")
    row_b[0].metric("Total Profit", fmt_num(summary["total_profit"]))
    row_b[1].metric("ROI", fmt_pct(summary["roi"]))
    row_b[2].metric("Win Rate", fmt_pct(summary["win_rate"]))
    row_b[3].metric("Avg Edge", fmt_pct(summary["avg_edge"]))

    if tracking.empty:
        st.info("No predictions have been saved yet.")
        return

    st.markdown(
        '<div class="small-label">History editor</div>',
        unsafe_allow_html=True,
    )
    edited_tracking = st.data_editor(
        tracking,
        width="stretch",
        num_rows="dynamic",
        disabled=[
            "Result",
            "Profit",
            "FixtureID",
            "League",
            "Referee",
            "PredictedCards",
            "ModelProbability",
            "MarketProbability",
            "ExpectedValue",
            "OddsSource",
            "SavedAt",
        ],
        column_order=TRACKING_COLUMNS,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "Pick": st.column_config.TextColumn("Pick"),
            "Result": st.column_config.SelectboxColumn(
                "Result", options=["PENDING", "WIN", "LOSS", "PUSH"]
            ),
            "Line": st.column_config.NumberColumn("Line", step=0.5),
            "Odds": st.column_config.NumberColumn("Odds", step=0.01),
            "Stake": st.column_config.NumberColumn("Stake", step=0.25),
            "Edge": st.column_config.NumberColumn("Edge", format="%.3f"),
            "Confidence": st.column_config.TextColumn("Confidence"),
            "FinalCards": st.column_config.NumberColumn(
                "Final Cards", min_value=0, step=1
            ),
            "Profit": st.column_config.NumberColumn("Profit", format="%.2f"),
            "PredictedCards": st.column_config.NumberColumn(
                "Predicted Cards", format="%.2f"
            ),
            "ModelProbability": st.column_config.NumberColumn(
                "Model Probability", format="percent"
            ),
            "MarketProbability": st.column_config.NumberColumn(
                "Market Probability", format="percent"
            ),
            "ExpectedValue": st.column_config.NumberColumn(
                "Expected Value", format="percent"
            ),
            "OddsSource": st.column_config.TextColumn("Odds Source"),
        },
        key="bet_tracking_editor",
    )

    evaluated_tracking = evaluate_tracking(edited_tracking)
    live_summary = calculate_summary_for_tracking(evaluated_tracking)

    section_header("Live Summary", kicker="Projected")
    p1, p2, p3, p4 = st.columns(4, gap="small")
    p1.metric("Projected Profit", fmt_num(live_summary["total_profit"]))
    p2.metric("Projected ROI", fmt_pct(live_summary["roi"]))
    p3.metric("Projected Win Rate", fmt_pct(live_summary["win_rate"]))
    p4.metric("Avg Edge", fmt_pct(live_summary["avg_edge"]))

    roi_by_confidence = pd.DataFrame(
        [
            {"Confidence": confidence, "ROI": roi}
            for confidence, roi in live_summary["roi_by_confidence"].items()
        ]
    )
    if not roi_by_confidence.empty:
        roi_by_confidence["ROI"] = roi_by_confidence["ROI"].map(
            lambda roi: f"{roi:.1%}"
        )
        st.dataframe(roi_by_confidence, width="stretch")

    with st.expander("Calculated results preview", expanded=False):
        st.dataframe(evaluated_tracking, width="stretch")

    if st.button("Save Final Cards", type="primary"):
        save_tracking_edits(evaluated_tracking)
        st.success("Bet tracker saved.")
        st.rerun()


def render_ultra_value(ultra_top_bets):
    st.subheader("Ultra Value")
    st.caption(
        "Ultra Value = research-backed strict filter: predicted cards <= 2.7, recent card intensity <= 3.0, over 4.5 probability < 30%, high confidence, and value edge > 5%."
    )

    if ultra_top_bets.empty:
        st.info("No Ultra Value matches available for today or future matches.")
        return

    for _, row in ultra_top_bets.iterrows():
        match_card(row)


def render_player_card_risk():
    section_header(
        "Player Card Risk",
        "Batch heuristic rankings from saved profiles and lineups. "
        "For live fixtures, use Live Match Builder.",
        kicker="Batch risk",
    )

    risks = load_player_risks()
    if risks.empty:
        st.info(
            "No player card risks available yet. Add rows to data/player_profiles.csv and data/lineups.csv, then run predictions."
        )
        return

    display_columns = ["Player", "Team", "Position", "RiskScore", "RiskTier"]
    for (match_date, home_team, away_team), match_risks in risks.groupby(
        ["Date", "HomeTeam", "AwayTeam"], dropna=False
    ):
        date_text = match_date.strftime("%m/%d/%Y") if pd.notna(match_date) else "N/A"
        st.markdown(f"### {home_team} vs {away_team}")
        st.caption(date_text)
        st.dataframe(match_risks[display_columns], width="stretch")



@st.cache_data(ttl=300)
def load_live_fixtures(selected_date):
    """Load and cache API fixtures for five minutes."""
    return get_normalized_fixtures_by_date(selected_date)


@st.cache_data(ttl=300)
def load_live_fixture_details(fixture_id):
    """Load and cache referee and lineup data for five minutes."""
    return get_fixture_lineup_data(fixture_id)


def render_live_match_builder():
    section_header(
        "Live Match Builder",
        "Select a date, league, and fixture to retrieve match details, "
        "referee information, and confirmed lineups automatically.",
        kicker="Live workflow",
        featured=True,
    )

    use_test_fixture = st.checkbox(
        "Use test fixture",
        key="use_test_live_fixture",
        help="Development only: use a local Arsenal vs Chelsea fixture without calling API-Football.",
    )

    if use_test_fixture:
        st.warning(
            "TEST DATA: Arsenal vs Chelsea is loaded locally. "
            "No fixture or lineup API requests will be made."
        )
        fixtures = [TEST_LIVE_FIXTURE.copy()]
    else:
        selected_date = st.date_input(
            "Fixture date",
            value=date.today(),
            key="live_fixture_date",
        )

        try:
            fixtures = load_live_fixtures(selected_date)
        except APIFootballError as error:
            st.error(f"Could not load fixtures: {error}")
            return

    if not fixtures:
        st.info("No fixtures were found for this date.")
        return

    leagues = sorted(
        {
            fixture.get("league")
            for fixture in fixtures
            if fixture.get("league")
        }
    )

    selector_left, selector_right = st.columns(2, gap="medium")
    with selector_left:
        selected_league = st.selectbox(
            "League",
            options=["All leagues"] + leagues,
            key="live_fixture_league",
        )
    with selector_right:
        filtered_fixtures = fixtures
        if selected_league != "All leagues":
            filtered_fixtures = [
                fixture
                for fixture in fixtures
                if fixture.get("league") == selected_league
            ]

        if not filtered_fixtures:
            st.info("No fixtures are available for the selected league.")
            return

        fixture_options = {
            get_fixture_label(fixture): fixture
            for fixture in filtered_fixtures
        }

        selected_fixture_label = st.selectbox(
            "Fixture",
            options=list(fixture_options.keys()),
            key="live_fixture_selection",
        )

    selected_fixture = fixture_options[selected_fixture_label]
    fixture_id = selected_fixture.get("fixture_id")

    if not fixture_id:
        st.warning("This fixture does not have a valid API fixture ID.")
        return

    if use_test_fixture:
        fixture_data = {
            "fixture": selected_fixture,
            "referee": selected_fixture.get("referee"),
            "lineups": TEST_LIVE_LINEUPS,
            "lineups_available": True,
        }
    else:
        try:
            fixture_data = load_live_fixture_details(fixture_id)
        except APIFootballError as error:
            st.error(f"Could not load fixture details: {error}")
            return

    fixture = fixture_data.get("fixture", {})
    referee = fixture_data.get("referee")
    lineup_rows = fixture_data.get("lineups", [])
    lineups_available = fixture_data.get("lineups_available", False)

    section_header("Match Details", kicker="Fixture")

    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("Home Team", fixture.get("home_team") or "Unknown")
    c2.metric("Away Team", fixture.get("away_team") or "Unknown")
    c3.metric("League", fixture.get("league") or "Unknown")
    c4.metric("Status", fixture.get("status") or "Unknown")

    referee_badge = (
        badge("REFEREE LOADED", "good")
        if referee
        else badge("NO REFEREE YET", "warn")
    )
    lineup_badge = (
        badge("LINEUPS CONFIRMED", "good")
        if lineups_available
        else badge("LINEUPS NOT RELEASED", "warn")
    )

    st.markdown(
        f"""
        <div class="panel panel-accent">
            <div class="small-label">Selected Fixture</div>
            <h3 style="margin: 6px 0 10px;">{fixture.get("home_team", "Unknown")} vs {fixture.get("away_team", "Unknown")}</h3>
            <div style="margin-top: 8px;">
                {referee_badge}
                {lineup_badge}
            </div>
            <div class="reason-box">
                <b>Referee:</b> {referee or "Not assigned or not yet available"}<br>
                <b>Kickoff:</b> {fixture.get("date") or "Unknown"}<br>
                <b>Country:</b> {fixture.get("country") or "Unknown"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Generate Match Prediction",
        type="primary",
        key=f"generate_live_prediction_{fixture_id}",
    ):
        with st.spinner("Training models and generating prediction..."):
            try:
                prediction = generate_live_prediction(fixture)
            except LivePredictionError as error:
                st.error(f"Could not generate prediction: {error}")
            else:
                st.session_state["live_prediction"] = prediction

    prediction = st.session_state.get("live_prediction")
    if prediction and prediction.get("fixture_id") == fixture_id:
        section_header(
            "Match Prediction",
            "Model output for the selected fixture.",
            kicker="Live model",
            featured=True,
        )
        render_live_prediction_summary(prediction)

        section_header(
            "Card Market Value",
            "Compare model probabilities against verified card-market odds.",
            kicker="Betting value",
            featured=True,
        )
        st.warning(
            "Entered odds must be for total match cards Over/Under 4.5, not goals. "
            "Generic soccer totals odds are not card-market odds."
        )
        odds_available = use_test_fixture or st.checkbox(
            "I have verified total-match card odds",
            key=f"verified_card_odds_{fixture_id}",
        )
        if not odds_available:
            calculate_live_betting_value(
                prediction["over_4_5_probability"],
                prediction["under_4_5_probability"],
                None,
                None,
                UNAVAILABLE_ODDS,
            )
            st.info(
                "Card-market odds are unavailable. No live betting recommendation "
                "will be made without verified prices."
            )
        else:
            source = MANUAL_TEST_ODDS if use_test_fixture else VERIFIED_CARD_MARKET
            st.caption(f"Odds source: {source}")
            odds_columns = st.columns(2, gap="medium")
            if use_test_fixture:
                over_odds = odds_columns[0].number_input(
                    "Over 4.5 decimal card odds",
                    min_value=1.01,
                    value=2.00,
                    step=0.01,
                    key=f"over_card_odds_{fixture_id}",
                )
                under_odds = odds_columns[1].number_input(
                    "Under 4.5 decimal card odds",
                    min_value=1.01,
                    value=1.80,
                    step=0.01,
                    key=f"under_card_odds_{fixture_id}",
                )
            else:
                over_odds = odds_columns[0].text_input(
                    "Over 4.5 decimal card odds",
                    placeholder="Enter verified card odds",
                    key=f"over_card_odds_{fixture_id}",
                )
                under_odds = odds_columns[1].text_input(
                    "Under 4.5 decimal card odds",
                    placeholder="Enter verified card odds",
                    key=f"under_card_odds_{fixture_id}",
                )
            stake = st.number_input(
                "Stake",
                min_value=0.0,
                value=1.0,
                step=0.5,
                key=f"card_value_stake_{fixture_id}",
            )
            if not use_test_fixture and (not over_odds.strip() or not under_odds.strip()):
                st.info("Enter both verified card-market prices to calculate value.")
                value = None
            else:
                try:
                    value = calculate_live_betting_value(
                        prediction["over_4_5_probability"],
                        prediction["under_4_5_probability"],
                        over_odds,
                        under_odds,
                        source,
                        stake,
                    )
                except ValueError as error:
                    st.error(str(error))
                    value = None
            if value is not None:
                st.caption(
                    "Bookmaker margin / overround: "
                    f"{fmt_pct(value['bookmaker_margin'])}"
                )
                render_betting_value_sides(value, stake)
                recommendation = value["recommendation"]
                if recommendation:
                    if st.button(
                        "Save to Prediction History",
                        key=f"save_live_prediction_{fixture_id}",
                    ):
                        try:
                            save_result = save_live_prediction(
                                fixture,
                                prediction,
                                value,
                                stake,
                            )
                        except ValueError as error:
                            st.error(str(error))
                        else:
                            if save_result["saved"]:
                                st.success("Prediction saved to history.")
                            else:
                                st.info(
                                    "This fixture, pick, and line already exist "
                                    "in Prediction History."
                                )

    if not lineups_available or not lineup_rows:
        st.info(
            "Confirmed lineups are not available yet. "
            "They are usually released shortly before kickoff."
        )
        return

    lineup_dataframe = pd.DataFrame(lineup_rows)
    starters = lineup_dataframe[
        lineup_dataframe["lineup_type"] == "Starter"
    ].copy()
    substitutes = lineup_dataframe[
        lineup_dataframe["lineup_type"] == "Substitute"
    ].copy()

    section_header("Confirmed Lineups", kicker="Squads")

    home_team = fixture.get("home_team")
    away_team = fixture.get("away_team")
    home_column, away_column = st.columns(2, gap="medium")

    with home_column:
        st.markdown(f"#### {home_team}")
        home_starters = starters[starters["team"] == home_team]
        if home_starters.empty:
            st.info("No starters available.")
        else:
            st.dataframe(
                home_starters[
                    ["number", "player", "position", "formation", "grid"]
                ],
                width="stretch",
                hide_index=True,
            )

        st.markdown("##### Substitutes")
        home_substitutes = substitutes[substitutes["team"] == home_team]
        if home_substitutes.empty:
            st.info("No substitutes available.")
        else:
            st.dataframe(
                home_substitutes[
                    ["number", "player", "position"]
                ],
                width="stretch",
                hide_index=True,
            )

    with away_column:
        st.markdown(f"#### {away_team}")
        away_starters = starters[starters["team"] == away_team]
        if away_starters.empty:
            st.info("No starters available.")
        else:
            st.dataframe(
                away_starters[
                    ["number", "player", "position", "formation", "grid"]
                ],
                width="stretch",
                hide_index=True,
            )

        st.markdown("##### Substitutes")
        away_substitutes = substitutes[substitutes["team"] == away_team]
        if away_substitutes.empty:
            st.info("No substitutes available.")
        else:
            st.dataframe(
                away_substitutes[
                    ["number", "player", "position"]
                ],
                width="stretch",
                hide_index=True,
            )

    section_header(
        "Live Player Card Risk",
        "CSV profiles are used first. Fetching is manual and limited to "
        "5 unmatched starters per action.",
        kicker="Player risk",
    )
    if not prediction or prediction.get("fixture_id") != fixture_id:
        st.info(
            "Generate the match prediction to score confirmed starters."
        )
        return

    fetch_missing_profiles = st.button(
        "Fetch Missing Player Profiles",
        key=f"fetch_missing_player_profiles_{fixture_id}",
    )

    try:
        live_risks = generate_live_player_card_risks(
            fixture,
            lineup_rows,
            prediction,
            fetch_missing=fetch_missing_profiles,
            max_api_fetches=5,
        )
    except PlayerStatsAPIError as error:
        st.error(f"Could not load player profile cache: {error}")
        return

    counts = live_risks.attrs.get("profile_counts", {})
    source_columns = st.columns(4, gap="small")
    source_columns[0].metric("CSV Profiles", counts.get("csv", 0))
    source_columns[1].metric("Cached API Profiles", counts.get("cache", 0))
    source_columns[2].metric("Live API Profiles", counts.get("live_api", 0))
    source_columns[3].metric("Unmatched", counts.get("unmatched", 0))

    if fetch_missing_profiles:
        st.caption(
            f"API profile requests used this action: "
            f"{live_risks.attrs.get('api_fetches', 0)} / 5"
        )
        for fetch_error in live_risks.attrs.get("fetch_errors", []):
            st.warning(fetch_error)
    matched_risks = live_risks[live_risks["ProfileMatched"]].copy()
    unmatched_risks = live_risks[~live_risks["ProfileMatched"]].copy()

    if matched_risks.empty:
        st.warning(
            "No confirmed starters matched data/player_profiles.csv. "
            "No player risk scores are available for this fixture."
        )
    else:
        st.markdown("#### Top Five Matched Starters")
        st.dataframe(
            matched_risks.head(5),
            width="stretch",
            hide_index=True,
        )
        if len(matched_risks) > 5:
            with st.expander("Other matched starters"):
                st.dataframe(
                    matched_risks.iloc[5:],
                    width="stretch",
                    hide_index=True,
                )

    if not unmatched_risks.empty:
        with st.expander(
            f"Unmatched starters ({len(unmatched_risks)})"
        ):
            st.caption(
                "These players were not assigned a risk score because no "
                "matching player-and-team profile was found."
            )
            st.dataframe(
                unmatched_risks,
                width="stretch",
                hide_index=True,
            )

predictions, top_bets, ultra_top_bets = load_data()

st.markdown(
    "<h1>Yellow Card Predictions</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>Machine learning-powered football yellow card predictions using referee tendencies, player risk analysis, betting odds, and historical match data.</div>",
    unsafe_allow_html=True,
)

high_conf = predictions[predictions["confidence"] == "HIGH CONFIDENCE"].copy()
missing_ref = predictions[predictions["confidence"] != "HIGH CONFIDENCE"].copy()
if "OddsSource" in predictions.columns:
    odds_source = predictions["OddsSource"].fillna("DEFAULT ODDS").astype(str).str.strip().str.upper()
else:
    odds_source = pd.Series("DEFAULT ODDS", index=predictions.index)
api_odds_count = int((odds_source == "API ODDS").sum())
manual_odds_count = int((odds_source == "MANUAL ODDS").sum())
default_odds_count = int((odds_source == "DEFAULT ODDS").sum())
referee_supported_count = int(
    predictions["Referee"].apply(lambda value: not is_missing_referee(value)).sum()
)

st.caption(
    f"Overview · {len(predictions)} upcoming · {len(high_conf)} high confidence · "
    f"{len(missing_ref)} missing ref · {len(top_bets)} top bets · "
    f"{len(ultra_top_bets)} ultra value"
)

with st.expander("Overview metrics", expanded=False):
    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    m1.metric("Upcoming Matches", len(predictions))
    m2.metric("High Confidence", len(high_conf))
    m3.metric("Missing Ref Data", len(missing_ref))
    m4.metric("Top Bets", len(top_bets))
    m5.metric("Ultra Value", len(ultra_top_bets))

    s1, s2, s3, s4 = st.columns(4, gap="small")
    s1.metric("API Odds Fixtures", api_odds_count)
    s2.metric("Manual Odds Fixtures", manual_odds_count)
    s3.metric("Default Odds Fixtures", default_odds_count)
    s4.metric("Referee-Supported Fixtures", referee_supported_count)

tab_live, tab_predictions, tab_history, tab_players = st.tabs(
    [
        "Live Match Builder",
        "Predictions",
        "Prediction History",
        "Player Card Risk",
    ]
)

with tab_live:
    render_live_match_builder()

with tab_predictions:
    render_predictions_hub(
        predictions,
        top_bets,
        ultra_top_bets,
        high_conf,
        missing_ref,
    )

with tab_history:
    render_bet_tracker()

with tab_players:
    render_player_card_risk()
