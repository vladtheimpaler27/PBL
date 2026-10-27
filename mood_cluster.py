import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from recommender import df

embeddings = np.load("embeddings.npy")

# Group songs into 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(embeddings)
np.save("clusters.npy", df["cluster"].values)

# What does each cluster sound like?
cols = ["energy", "valence", "danceability", "acousticness", "tempo"]
summary = df.groupby("cluster")[cols].mean().round(2)
summary["songs"] = df["cluster"].value_counts().sort_index()
print(summary)

for c in range(5):
    print(f"\nCluster {c} examples:")
    print(df[df.cluster == c][["track_name", "track_artist", "playlist_genre"]]
          .head(5).to_string(index=False))

# Plot all songs in 2D, coloured by cluster
coords = PCA(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], c=df["cluster"], cmap="tab10", s=8)
plt.title("Songs grouped into mood clusters")
plt.savefig("clusters.png", dpi=150)
print("\nSaved clusters.png")

