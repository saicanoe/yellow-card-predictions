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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.20), transparent 35%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 30%),
            #020617;
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1500px;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.06em;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    h2, h3 {
        color: #f8fafc;
        letter-spacing: -0.03em;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(2, 6, 23, 0.95));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.35);
    }

    .match-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.72));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 16px 50px rgba(0,0,0,0.28);
    }

    .badge {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
    }

    .good { background: rgba(34,197,94,.15); color: #86efac; border: 1px solid rgba(34,197,94,.35); }
    .warn { background: rgba(245,158,11,.14); color: #fcd34d; border: 1px solid rgba(245,158,11,.35); }
    .bad { background: rgba(239,68,68,.14); color: #fca5a5; border: 1px solid rgba(239,68,68,.35); }
    .neutral { background: rgba(59,130,246,.14); color: #93c5fd; border: 1px solid rgba(59,130,246,.35); }

    .small-label {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 700;
    }

    .big-number {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        margin-top: -5px;
    }

    .reason-box {
        background: rgba(2, 6, 23, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 15px;
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,.15);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15,23,42,.88);
        border-radius: 999px;
        padding: 10px 18px;
        border: 1px solid rgba(148,163,184,.18);
        color: #cbd5e1;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        color: white;
        border: none;
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

    st.markdown("<div class='match-card'>", unsafe_allow_html=True)

    top_left, top_right = st.columns([2.2, 1])

    with top_left:
        st.markdown(
            f"""
            <div class="small-label">{row.get('Div', '')} · {row['Date'].strftime('%m/%d/%Y')}</div>
            <h3 style="margin-top: 6px;">{row['HomeTeam']} vs {row['AwayTeam']}</h3>
            {badge(signal, signal_kind)} {badge(confidence, confidence_kind)} {badge('VALUE: ' + value_bet, value_kind)}
            <div style="margin-top: 10px;">{data_source_badges(row)}</div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        st.markdown(
            f"""
            <div class="small-label">Referee</div>
            <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;">{referee_display(row.get('Referee'))}</div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='small-label'>Predicted Cards</div><div class='big-number'>{fmt_num(row['predicted_cards'])}</div></div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='small-label'>Under 4.5 Model Prob</div><div class='big-number'>{fmt_pct(row.get('under_model_prob'))}</div></div>",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"<div class='metric-card'><div class='small-label'>Book Prob</div><div class='big-number'>{fmt_pct(row.get('under_book_prob'))}</div></div>",
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"<div class='metric-card'><div class='small-label'>Value Edge</div><div class='big-number'>{fmt_pct(row.get('value_edge'))}</div></div>",
            unsafe_allow_html=True,
        )

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

    st.markdown("</div>", unsafe_allow_html=True)


def render_bet_tracker():
    summary = calculate_summary()
    tracking = load_tracking_data()

    st.subheader("Bet Tracker")

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Total Bets", summary["total_bets"])
    c2.metric("Wins", summary["wins"])
    c3.metric("Losses", summary["losses"])
    c4.metric("Pending", summary["pending"])
    c5.metric("Total Profit", fmt_num(summary["total_profit"]))
    c6.metric("ROI", fmt_pct(summary["roi"]))
    c7.metric("Win Rate", fmt_pct(summary["win_rate"]))
    c8.metric("Avg Edge", fmt_pct(summary["avg_edge"]))

    if tracking.empty:
        st.info("No bets logged yet. Run predictions to add top bets automatically.")
        return

    st.caption(
        "Enter FinalCards after a match finishes. Result and Profit are calculated automatically from Pick, Line, Odds, and Stake."
    )
    edited_tracking = st.data_editor(
        tracking,
        use_container_width=True,
        num_rows="dynamic",
        disabled=["Result", "Profit"],
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
            "FinalCards": st.column_config.NumberColumn("Final Cards", step=1),
            "Profit": st.column_config.NumberColumn("Profit", format="%.2f"),
        },
        key="bet_tracking_editor",
    )

    evaluated_tracking = evaluate_tracking(edited_tracking)
    live_summary = calculate_summary_for_tracking(evaluated_tracking)

    st.markdown("#### Live Summary")
    p1, p2, p3, p4 = st.columns(4)
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
        st.dataframe(roi_by_confidence, use_container_width=True)

    st.markdown("#### Calculated Results Preview")
    st.dataframe(evaluated_tracking, use_container_width=True)

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
    st.subheader("Player Card Risk")
    st.caption(
        "Early heuristic model: this ranks manual player profiles and lineups using position, card rate, and match context. It is not a trained player model yet."
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
        st.dataframe(match_risks[display_columns], use_container_width=True)



@st.cache_data(ttl=300)
def load_live_fixtures(selected_date):
    """Load and cache API fixtures for five minutes."""
    return get_normalized_fixtures_by_date(selected_date)


@st.cache_data(ttl=300)
def load_live_fixture_details(fixture_id):
    """Load and cache referee and lineup data for five minutes."""
    return get_fixture_lineup_data(fixture_id)


def render_live_match_builder():
    st.subheader("Live Match Builder")
    st.caption(
        "Select a date, league, and fixture to retrieve match details, "
        "referee information, and confirmed lineups automatically."
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

    selected_league = st.selectbox(
        "League",
        options=["All leagues"] + leagues,
        key="live_fixture_league",
    )

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

    st.markdown("### Match Details")

    c1, c2, c3, c4 = st.columns(4)
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
        <div class="match-card">
            <div class="small-label">Selected Fixture</div>
            <h3>{fixture.get("home_team", "Unknown")} vs {fixture.get("away_team", "Unknown")}</h3>
            <div style="margin-top: 12px;">
                {referee_badge}
                {lineup_badge}
            </div>
            <div style="margin-top: 16px;">
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
        st.markdown("### Match Prediction")

        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Predicted Total Cards",
            fmt_num(prediction["predicted_cards"]),
        )
        p2.metric(
            "Over 4.5 Probability",
            fmt_pct(prediction["over_4_5_probability"]),
        )
        p3.metric(
            "Under 4.5 Probability",
            fmt_pct(prediction["under_4_5_probability"]),
        )

        profile_found = prediction["referee_profile_found"]
        profile_badge = (
            badge("REFEREE PROFILE FOUND", "good")
            if profile_found
            else badge("LEAGUE AVERAGE PROFILE", "warn")
        )
        referee_name = (
            prediction.get("referee_model")
            or "No referee assigned"
        )

        st.markdown(
            f"""
            <div class="match-card">
                <div class="small-label">Prediction Inputs</div>
                <div style="margin: 10px 0 16px;">{profile_badge}</div>
                <div class="reason-box">
                    <b>Referee profile status:</b> {referee_name}<br>
                    <b>Historical team mapping:</b><br>
                    {prediction["home_team_api"]} &rarr; {prediction["home_team_model"]}<br>
                    {prediction["away_team_api"]} &rarr; {prediction["away_team_model"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Card Market Value")
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
            odds_columns = st.columns(2)
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
                rows = []
                for side, metrics in value["sides"].items():
                    rows.append(
                        {
                            "Side": side,
                            "Model Probability": fmt_pct(metrics["model_probability"]),
                            "Decimal Odds": f"{metrics['decimal_odds']:.2f}",
                            "No-Vig Market Probability": fmt_pct(
                                metrics["no_vig_implied_probability"]
                            ),
                            "Edge": fmt_pct(metrics["probability_edge"]),
                            "Expected Value": fmt_pct(
                                metrics["expected_value_per_unit"]
                            ),
                            f"Expected Profit ({stake:g} stake)": (
                                f"{metrics['expected_profit']:+.2f}"
                            ),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )
                recommendation = value["recommendation"]
                if recommendation:
                    st.success(f"Recommendation: {recommendation}")
                else:
                    st.info("No positive EV")

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

    st.markdown("### Confirmed Lineups")

    home_team = fixture.get("home_team")
    away_team = fixture.get("away_team")
    home_column, away_column = st.columns(2)

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
                use_container_width=True,
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
                use_container_width=True,
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
                use_container_width=True,
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
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### Live Player Card Risk")
    if not prediction or prediction.get("fixture_id") != fixture_id:
        st.info(
            "Generate the match prediction to score confirmed starters."
        )
        return

    st.caption(
        "CSV profiles are used first. Fetching is manual and limited to "
        "5 unmatched starters per action."
    )
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
    source_columns = st.columns(4)
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
            use_container_width=True,
            hide_index=True,
        )
        if len(matched_risks) > 5:
            with st.expander("Other matched starters"):
                st.dataframe(
                    matched_risks.iloc[5:],
                    use_container_width=True,
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
                use_container_width=True,
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

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Upcoming Matches", len(predictions))
m2.metric("High Confidence", len(high_conf))
m3.metric("Missing Ref Data", len(missing_ref))
m4.metric("Top Bets", len(top_bets))
m5.metric("Ultra Value", len(ultra_top_bets))

s1, s2, s3, s4 = st.columns(4)
s1.metric("API Odds Fixtures", api_odds_count)
s2.metric("Manual Odds Fixtures", manual_odds_count)
s3.metric("Default Odds Fixtures", default_odds_count)
s4.metric("Referee-Supported Fixtures", referee_supported_count)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Top Bets",
        "Ultra Value",
        "With Ref Data",
        "Without Ref Data",
        "All Predictions",
        "Bet Tracker",
        "Player Card Risk",
        "Live Match Builder",
    ]
)

with tab1:
    st.subheader("Top Bets")
    if top_bets.empty:
        st.info("No top bets available for today or future matches.")
    else:
        for _, row in top_bets.iterrows():
            match_card(row)

with tab2:
    render_ultra_value(ultra_top_bets)

with tab3:
    st.subheader("Matches With Referee Data")
    if high_conf.empty:
        st.info("No high-confidence matches available.")
    else:
        for _, row in high_conf.iterrows():
            match_card(row)

with tab4:
    st.subheader("Matches Without Referee Data")
    st.caption(
        "These matches are shown separately because the model is using fallback average referee values."
    )
    if missing_ref.empty:
        st.success("No matches are missing referee data.")
    else:
        for _, row in missing_ref.iterrows():
            match_card(row)

with tab5:
    st.subheader("All Upcoming Predictions")
    st.dataframe(predictions, use_container_width=True)

with tab6:
    render_bet_tracker()

with tab7:
    render_player_card_risk()

with tab8:
    render_live_match_builder()
