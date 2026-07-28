import pandas as pd
import glob

files = glob.glob("data/SP1_*.csv")

df_list = []
for file in files:
    temp = pd.read_csv(file)
    temp["source_file"] = file
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print("Columns:")
print(df.columns.tolist())

if "Referee" not in df.columns:
    print("No Referee column found in Spanish files.")
else:
    refs = (
        df["Referee"]
        .dropna()
        .value_counts()
        .reset_index()
    )

    refs.columns = ["Referee", "matches"]

    print("\nSpanish refs found:")
    print(refs.to_string(index=False))