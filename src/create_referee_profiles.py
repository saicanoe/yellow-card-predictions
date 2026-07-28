import pandas as pd
import glob

files = [f for f in glob.glob("data/*.csv") if "upcoming" not in f.lower()]

df_list = []

for file in files:
    temp = pd.read_csv(file)
    temp["source_file"] = file
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print("Columns found:")
print(df.columns.tolist())

# Make sure required columns exist
required = ["Referee", "HY", "AY", "HR", "AR"]

missing = [col for col in required if col not in df.columns]

if missing:
    print("Missing columns:", missing)
    print("Your CSVs may not have referee data.")
    exit()

# Create card totals
df["total_yellows"] = df["HY"] + df["AY"]
df["total_reds"] = df["HR"] + df["AR"]
df["total_cards"] = df["total_yellows"] + df["total_reds"]

# Drop rows without referee
df = df.dropna(subset=["Referee"])

# Referee profiles
ref_profiles = df.groupby("Referee").agg(
    matches=("Referee", "count"),
    avg_yellows=("total_yellows", "mean"),
    avg_reds=("total_reds", "mean"),
    avg_total_cards=("total_cards", "mean"),
    over_3_5_rate=("total_cards", lambda x: (x > 3.5).mean()),
    over_4_5_rate=("total_cards", lambda x: (x > 4.5).mean()),
    over_5_5_rate=("total_cards", lambda x: (x > 5.5).mean()),
    home_yellows_avg=("HY", "mean"),
    away_yellows_avg=("AY", "mean")
).reset_index()

# Sort highest-card refs first
ref_profiles = ref_profiles.sort_values("avg_total_cards", ascending=False)

# Save
ref_profiles.to_csv("data/referees.csv", index=False)

print("Created data/referees.csv")
print(ref_profiles.head(15))
