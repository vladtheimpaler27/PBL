import pandas as pd

high = pd.read_csv("high_popularity_spotify_data.csv")
low = pd.read_csv("low_popularity_spotify_data.csv")

df = pd.concat([high, low], ignore_index=True)

print("Rows and columns:", df.shape)
print()
print("Columns:")
for col in df.columns:
    print(" -", col, "|", df[col].dtype, "|", df[col].nunique(), "unique values")
print()
print(df.head(3).T)
