import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recommender import df, feature_matrix

embeddings = np.load("embeddings.npy")
genres = df["playlist_genre"].values

def genre_match(vectors, k=10, sample=300):
    rng = np.random.default_rng(42)
    picks = rng.choice(len(df), sample, replace=False)
    total = 0
    for i in picks:
        scores = cosine_similarity(vectors[i].reshape(1, -1), vectors)[0]
        top = scores.argsort()[::-1][1:k + 1]
        total += (genres[top] == genres[i]).mean()
    return total / sample

print("Basic  genre match:", round(genre_match(feature_matrix), 3))
print("AI     genre match:", round(genre_match(embeddings), 3))