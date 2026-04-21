import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Movie Analytics System", layout="wide")

# ------------------------------
# SESSION STATE
# ------------------------------
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "movies.csv",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )

    df.columns = df.columns.str.strip()

    # Fix numeric columns
    df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
    df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
    df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")

    # Fix text
    df["Genre"] = df["Genre"].fillna("")
    df["Overview"] = df.get("Overview", "").fillna("")
    df["Title"] = df["Title"].fillna("Unknown")

    df.dropna(subset=["Vote_Average", "Popularity"], inplace=True)

    return df

df = load_data()

if df.empty:
    st.error("Data failed to load")
    st.stop()

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
st.sidebar.title("🎬 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Dashboard", "📈 Trends", "🤖 Recommendations"]
)

# ------------------------------
# 🏠 HOME PAGE
# ------------------------------
if page == "🏠 Home":

    st.title("🎬 Movie Analytics & Recommendation System")

    # 🔍 Smart Search + Filters
    search = st.text_input("🔍 Search Movie")

    col1, col2 = st.columns(2)

    with col1:
        min_rating = st.slider("Min Rating", 0.0, 10.0, 5.0)

    with col2:
        genre_filter = st.selectbox("Genre", ["All"] + list(df["Genre"].unique()))

    filtered = df.copy()

    if search:
        filtered = filtered[filtered["Title"].str.contains(search, case=False, na=False)]

    if genre_filter != "All":
        filtered = filtered[filtered["Genre"] == genre_filter]

    filtered = filtered[filtered["Vote_Average"] >= min_rating]

    st.write(f"Results: {len(filtered)}")

    cols = st.columns(4)

    for i, (_, row) in enumerate(filtered.head(12).iterrows()):
        with cols[i % 4]:
            st.image(row["Poster_Url"], use_container_width=True)
            if st.button(row["Title"], key=i):
                st.session_state.selected_movie = row["Title"]

# ------------------------------
# 🎬 MOVIE DETAIL PAGE
# ------------------------------
if st.session_state.selected_movie:

    movie = df[df["Title"] == st.session_state.selected_movie].iloc[0]

    st.title(movie["Title"])

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(movie["Poster_Url"], use_container_width=True)

    with col2:
        st.write(f"⭐ Rating: {movie['Vote_Average']}")
        st.write(f"🔥 Popularity: {movie['Popularity']}")
        st.write(f"🎭 Genre: {movie['Genre']}")
        st.write(movie["Overview"])

    if st.button("⬅ Back"):
        st.session_state.selected_movie = None
        st.rerun()

# ------------------------------
# 📊 DASHBOARD PAGE
# ------------------------------
elif page == "📊 Dashboard":

    st.title("📊 Advanced Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Movies", len(df))
    col2.metric("Avg Rating", round(df["Vote_Average"].mean(), 2))
    col3.metric("Max Popularity", round(df["Popularity"].max(), 2))

    genre = st.selectbox("Genre", df["Genre"].unique())
    filtered_df = df[df["Genre"] == genre]

    # Plotly charts
    fig1 = px.histogram(filtered_df, x="Popularity")
    st.plotly_chart(fig1)

    fig2 = px.scatter(filtered_df, x="Vote_Average", y="Popularity", hover_data=["Title"])
    st.plotly_chart(fig2)

    top10 = filtered_df.sort_values("Popularity", ascending=False).head(10)
    fig3 = px.bar(top10, x="Title", y="Popularity")
    st.plotly_chart(fig3)

# ------------------------------
# 📈 TRENDS PAGE
# ------------------------------
elif page == "📈 Trends":

    st.title("📈 Trends & Insights")

    if "Release_Date" in df.columns:
        df["Year"] = pd.to_datetime(df["Release_Date"], errors="coerce").dt.year
        yearly = df["Year"].value_counts().sort_index()
        st.line_chart(yearly)

    st.subheader("Genre Distribution")
    genre_counts = df["Genre"].value_counts().head(10)

    fig, ax = plt.subplots()
    ax.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)

# ------------------------------
# 🤖 RECOMMENDATION PAGE
# ------------------------------
elif page == "🤖 Recommendations":

    st.title("🤖 Hybrid Recommendation System")

    df["combined"] = (df["Genre"] + " " + df["Overview"]).fillna("")

    cv = CountVectorizer(stop_words='english')
    matrix = cv.fit_transform(df["combined"])
    similarity = cosine_similarity(matrix)

    movie_list = df["Title"].values
    selected_movie = st.selectbox("Select Movie", movie_list)

    def recommend(movie):
        idx = df[df["Title"] == movie].index[0]
        sim_scores = similarity[idx]

        movie_indices = sim_scores.argsort()[-10:][::-1]

        hybrid_scores = []
        for i in movie_indices:
            score = (sim_scores[i] * 0.7) + (df.iloc[i]["Popularity"] * 0.3 / df["Popularity"].max())
            hybrid_scores.append((i, score))

        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)

        return [(df.iloc[i], score) for i, score in hybrid_scores[:5]]

    if st.button("Recommend"):

        st.write(f"Because you watched **{selected_movie}**")

        results = recommend(selected_movie)

        cols = st.columns(5)

        for i, (row, score) in enumerate(results):
            with cols[i]:
                st.image(row["Poster_Url"], use_container_width=True)
                st.caption(row["Title"])
                st.caption(f"{round(score*100,2)}% match")