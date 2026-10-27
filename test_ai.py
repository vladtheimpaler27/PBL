from recommender import df, recommend, recommend_ai

song = df.loc[0,"track_name"]
print("BASIC:\n",recommend(song))
print("\nAI :\n", recommend_ai(song))
