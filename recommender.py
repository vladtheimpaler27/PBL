import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load both files and join them
high = pd.read_csv("high_popularity_spotify_data.csv")
low = pd.read_csv("low_popularity_spotify_data.csv")
df = pd.concat([high, low], ignore_index=True)

features = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

# 2. Clean: remove missing values and duplicate songs
df = df.dropna(subset=["track_name", "track_artist", "playlist_genre"] + features)
df = df.drop_duplicates(subset=["track_id"])
df = df.drop_duplicates(subset=["track_name", "track_artist"]).reset_index(drop=True)

# 3. Scale audio features to 0-1
feature_matrix = MinMaxScaler().fit_transform(df[features])

# 4. Find a song by name (exact match first, then partial match)
def find_song(song_name):
    name = song_name.lower().strip()
    exact = df[df["track_name"].str.lower() == name]
    if not exact.empty:
        return exact.index[0]
    partial = df[df["track_name"].str.lower().str.contains(name, regex=False)]
    if not partial.empty:
        return partial.index[0]
    return None

def _top_similar(vectors, idx, n):
    scores = cosine_similarity(vectors[idx].reshape(1, -1), vectors)[0]
    top = scores.argsort()[::-1][1:n + 1]
    result = df.loc[top, ["track_name", "track_artist", "playlist_genre"]].copy()
    result["similarity"] = scores[top].round(3)
    return result

# 5. Basic recommender
def recommend(song_name, n=10):
    idx = find_song(song_name)
    if idx is None:
        return f"'{song_name}' not found."
    return _top_similar(feature_matrix, idx, n)

# 6. AI recommender (works after you run train_model.py)
def recommend_ai(song_name, n=10):
    if not os.path.exists("embeddings.npy"):
        return "Train the model first (python3 train_model.py)."
    embeddings = np.load("embeddings.npy")
    idx = find_song(song_name)
    if idx is None:
        return f"'{song_name}' not found."
    return _top_similar(embeddings, idx, n)

if __name__ == "__main__":
    print("Songs in dataset:", df.shape[0])
    print(df[["track_name", "track_artist"]].head(10))
    first_song = df.loc[0, "track_name"]
    print(f"\nSongs similar to '{first_song}':")
    print(recommend(first_song))