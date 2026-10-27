import os
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from recommender import df, recommend, recommend_ai
from bandit import LinUCB

st.title("🎵 Song Recommender!")

# ---------- Mood clusters (made by mood_clusters.py) ----------
has_clusters = os.path.exists("clusters.npy")
if has_clusters:
    df["cluster"] = np.load("clusters.npy")
    n_arms = int(df["cluster"].nunique())


# ---------- Helper: show a Spotify player for a song ----------
def play_song(track_id, name, artist):
    components.iframe(f"https://open.spotify.com/embed/track/{track_id}", height=152)
    query = quote_plus(f"{name} {artist}")
    c1, c2 = st.columns(2)
    c1.link_button("Open in Spotify", f"https://open.spotify.com/track/{track_id}")
    c2.link_button("Search on YouTube",
                   f"https://www.youtube.com/results?search_query={query}")


tab1, tab2, tab3 = st.tabs(["Similar songs", "Mood playlist", "Learn my taste"])

# ---------- Tab 1: similar songs ----------
with tab1:
    song = st.text_input("Type a song name", df.loc[0, "track_name"])
    method = st.radio("Method", ["Basic", "AI model"])
    n = st.slider("How many songs?", 5, 20, 10)

    if st.button("Recommend"):
        st.session_state.sim_result = (
            recommend(song, n) if method == "Basic" else recommend_ai(song, n)
        )

    result = st.session_state.get("sim_result")
    if isinstance(result, str):
        st.warning(result)
    elif result is not None:
        st.dataframe(result)
        pick = st.selectbox(
            "Play one of these",
            result.index.tolist(),
            format_func=lambda i: f"{result.loc[i, 'track_name']} - {result.loc[i, 'track_artist']}",
        )
        play_song(df.loc[pick, "track_id"],
                  df.loc[pick, "track_name"],
                  df.loc[pick, "track_artist"])

# ---------- Tab 2: mood playlist ----------
with tab2:
    if not has_clusters:
        st.warning("Run mood_clusters.py first.")
    else:
        mood = st.selectbox("Mood cluster", sorted(df["cluster"].unique()))
        size = st.slider("Playlist length", 5, 30, 15)

        if st.button("Make playlist"):
            pool = df[df["cluster"] == mood]
            st.session_state.playlist = pool.sample(min(size, len(pool)))

        playlist = st.session_state.get("playlist")
        if playlist is not None:
            st.dataframe(playlist[["track_name", "track_artist", "playlist_genre"]])
            pick2 = st.selectbox(
                "Play one of these",
                playlist.index.tolist(),
                format_func=lambda i: f"{playlist.loc[i, 'track_name']} - {playlist.loc[i, 'track_artist']}",
                key="playlist_pick",
            )
            play_song(df.loc[pick2, "track_id"],
                      df.loc[pick2, "track_name"],
                      df.loc[pick2, "track_artist"])

# ---------- Tab 3: learn my taste (bandit) ----------
with tab3:
    if not has_clusters:
        st.warning("Run mood_clusters.py first.")
    else:
        if "bandit" not in st.session_state:
            st.session_state.bandit = LinUCB(n_arms, d=2, alpha=0.5)
            st.session_state.pulls = np.zeros(n_arms)
            st.session_state.wins = np.zeros(n_arms)
            st.session_state.current = None

        st.slider("How energetic do you feel right now?", 0.0, 1.0, 0.5,
                  key="energy_pref")

        def pick_song():
            x = np.array([1.0, st.session_state.energy_pref])
            arm = st.session_state.bandit.choose(x)
            song_row = df[df["cluster"] == arm].sample(1).iloc[0]
            st.session_state.current = (arm, song_row["track_name"],
                                        song_row["track_artist"], x,
                                        song_row["track_id"])

        def give_feedback(reward):
            arm, _, _, x, _ = st.session_state.current
            st.session_state.bandit.update(arm, x, reward)
            st.session_state.pulls[arm] += 1
            if reward > 0:
                st.session_state.wins[arm] += 1
            pick_song()

        def reset():
            for k in ["bandit", "pulls", "wins", "current"]:
                st.session_state.pop(k, None)

        if st.session_state.current is None:
            st.button("Start", on_click=pick_song)
        else:
            arm, name, artist, _, tid = st.session_state.current
            st.subheader(f"{name} by {artist}")
            st.caption(f"Picked from mood cluster {arm}")
            play_song(tid, name, artist)

            c1, c2 = st.columns(2)
            c1.button("👍 Like", on_click=give_feedback, args=(1,))
            c2.button("👎 Dislike", on_click=give_feedback, args=(-1,))

            stats = pd.DataFrame({
                "cluster": range(n_arms),
                "shown": st.session_state.pulls.astype(int),
                "likes": st.session_state.wins.astype(int),
            })
            stats["like rate"] = (stats["likes"] /
                                  stats["shown"].clip(lower=1)).round(2)
            st.write("What the system has learned so far:")
            st.dataframe(stats, hide_index=True)
            st.button("Reset learning", on_click=reset)

        