import pandas as pd

FEATURE_COLUMNS = [
    "HomeTeamCode",
    "AwayTeamCode",
    "league_code",
    "home_cards_last5",
    "away_cards_last5",
    "total_cards_last5",
    "home_season_avg",
    "away_season_avg",
    "home_advantage",
    "avg_total_cards",
    "over_4_5_rate",
]


def prepare_training_data(raw_matches: pd.DataFrame, ref_stats: pd.DataFrame):
    """Create the same historical features used by the prototype script."""
    df = raw_matches

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["total_cards"] = df["HY"] + df["AY"]
    df["league_code"] = df["Div"].astype("category").cat.codes
    df["over_4_5"] = (df["total_cards"] > 4.5).astype(int)
    df["home_cards"] = df["HY"]
    df["away_cards"] = df["AY"]
    df = df.sort_values("Date")

    df["home_advantage"] = df["home_cards"] - df["away_cards"]
    df["home_season_avg"] = df.groupby("HomeTeam")["home_cards"].transform("mean")
    df["away_season_avg"] = df.groupby("AwayTeam")["away_cards"].transform("mean")

    df = df.merge(ref_stats, on="Referee", how="left").copy()

    df["home_cards_last5"] = df.groupby("HomeTeam")["home_cards"].transform(
        lambda cards: cards.shift().rolling(5).mean()
    )
    df["away_cards_last5"] = df.groupby("AwayTeam")["away_cards"].transform(
        lambda cards: cards.shift().rolling(5).mean()
    )
    df["total_cards_last5"] = df["home_cards_last5"] + df["away_cards_last5"]

    profiles = {
        "home_last5": df.groupby("HomeTeam")["home_cards"]
        .apply(lambda cards: cards.tail(5).mean())
        .to_dict(),
        "away_last5": df.groupby("AwayTeam")["away_cards"]
        .apply(lambda cards: cards.tail(5).mean())
        .to_dict(),
    }

    df = df.dropna(
        subset=[
            "HomeTeam",
            "AwayTeam",
            "home_cards_last5",
            "away_cards_last5",
            "avg_total_cards",
            "over_4_5_rate",
            "HY",
            "AY",
        ]
    ).copy()

    team_names = pd.concat([df["HomeTeam"], df["AwayTeam"]]).dropna().unique()
    team_to_code = {team: i for i, team in enumerate(team_names)}
    df["HomeTeamCode"] = df["HomeTeam"].map(team_to_code)
    df["AwayTeamCode"] = df["AwayTeam"].map(team_to_code)

    profiles["home_season"] = df.groupby("HomeTeam")["home_cards"].mean().to_dict()
    profiles["away_season"] = df.groupby("AwayTeam")["away_cards"].mean().to_dict()

    league_names = df[["Div", "league_code"]].drop_duplicates()
    league_to_code = dict(zip(league_names["Div"], league_names["league_code"]))

    return df, team_to_code, league_to_code, profiles


def add_upcoming_features(
    fixtures: pd.DataFrame,
    training_data: pd.DataFrame,
    ref_stats: pd.DataFrame,
    team_to_code: dict,
    league_to_code: dict,
    profiles: dict,
) -> pd.DataFrame:
    """Merge refs and build model-ready features for upcoming fixtures."""
    fixtures = fixtures.copy()

    fixtures = fixtures.merge(ref_stats, on="Referee", how="left")
    fixtures["avg_total_cards"] = fixtures["avg_total_cards"].fillna(
        training_data["avg_total_cards"].mean()
    )
    fixtures["over_4_5_rate"] = fixtures["over_4_5_rate"].fillna(
        training_data["over_4_5_rate"].mean()
    )

    upcoming_features = pd.DataFrame(
        {
            "HomeTeamCode": fixtures["HomeTeam"].map(team_to_code),
            "AwayTeamCode": fixtures["AwayTeam"].map(team_to_code),
            "league_code": fixtures["Div"].map(league_to_code),
            "home_cards_last5": fixtures["HomeTeam"].map(profiles["home_last5"]),
            "away_cards_last5": fixtures["AwayTeam"].map(profiles["away_last5"]),
            "home_season_avg": fixtures["HomeTeam"].map(profiles["home_season"]),
            "away_season_avg": fixtures["AwayTeam"].map(profiles["away_season"]),
        },
        index=fixtures.index,
    )
    upcoming_features["total_cards_last5"] = (
        upcoming_features["home_cards_last5"] + upcoming_features["away_cards_last5"]
    )
    upcoming_features["home_advantage"] = (
        upcoming_features["home_cards_last5"] - upcoming_features["away_cards_last5"]
    )

    fixtures = fixtures.join(upcoming_features)
    return fixtures.dropna(
        subset=[
            "HomeTeamCode",
            "AwayTeamCode",
            "home_cards_last5",
            "away_cards_last5",
        ]
    ).copy()
