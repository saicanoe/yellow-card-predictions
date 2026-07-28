import pandas as pd
ref_epl = pd.read_csv("data/referees.csv")
ref_spain = pd.read_csv("data/referees_spain.csv")
refs = pd.concat([ref_epl, ref_spain], ignore_index=True)
upcoming = pd.read_csv("data/upcoming_referees.csv")

matched = upcoming.merge(refs, on="Referee", how="left")

print("\nRefs you entered:")
print(upcoming["Referee"].unique())

print("\nMatched refs:")
print(
    matched[
        ["HomeTeam", "AwayTeam", "Referee", "matches", "avg_total_cards", "over_4_5_rate"]
    ].to_string(index=False)
)

print("\nUnknown refs:")
print(
    matched[matched["matches"].isna()][["HomeTeam", "AwayTeam", "Referee"]].to_string(index=False)
)