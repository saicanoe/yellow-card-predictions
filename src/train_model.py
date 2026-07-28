import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Load all CSV files (EPL + La Liga)
files = glob.glob("data/*.csv")

df_list = []
for file in files:
    df = pd.read_csv(file)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# Create total cards
df["total_cards"] = df["HY"] + df["AY"]

# Sort by date (VERY IMPORTANT)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.sort_values("Date")

# Create team card history
df["home_cards"] = df["HY"]
df["away_cards"] = df["AY"]

# Rolling averages (last 5 matches)
df["home_cards_last5"] = (
    df.groupby("HomeTeam")["home_cards"]
    .transform(lambda x: x.shift().rolling(5).mean())
)

df["away_cards_last5"] = (
    df.groupby("AwayTeam")["away_cards"]
    .transform(lambda x: x.shift().rolling(5).mean())
)

# Drop rows without enough history
df = df.dropna(subset=["home_cards_last5", "away_cards_last5"])

# Encode teams
df["HomeTeam"] = df["HomeTeam"].astype("category").cat.codes
df["AwayTeam"] = df["AwayTeam"].astype("category").cat.codes

# Features (PRE-MATCH ONLY)
X = df[["HomeTeam", "AwayTeam", "home_cards_last5", "away_cards_last5"]]
y = df["total_cards"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)

model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, preds)
print("MAE:", mae)

# Example prediction
sample = X_test.iloc[0:1]
pred_value = model.predict(sample)[0]

print("Predicted cards:", pred_value)

book_line = 4.5

if pred_value > book_line:
    print("Signal: OVER 4.5 cards")
else:
    print("Signal: UNDER 4.5 cards")