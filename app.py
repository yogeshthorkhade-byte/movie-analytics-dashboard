import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Movie Analytics Dashboard",
    layout="wide",
    page_icon="🎬"
)

# ------------------------------
# NETFLIX UI STYLE
# ------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #E50914;
}
.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# SESSION STATE (NAVIGATION)
# ------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "favorites" not in st.session_state:
    st.session_state.favorites = []

# ------------------------------
# LOAD DATA (SAFE)
# ------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "movies.csv",
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

        df.columns = df.columns.str.strip()

        df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
        df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
        df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")

        df.dropna(inplace=True)

        return df
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ Data failed to load")
    st.stop()

# ------------------------------
# ML MODEL (COSINE SIMILARITY)
# ------------------------------
df["combined"] = df["Genre"].astype(str) + " " + df.get("Overview", "").astype(str)

cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(df["combined"])
similarity = cosine_similarity(matrix)

def recommend(movie):
    idx = df[df["Title"] == movie].index[0]
    distances = similarity[idx]
    movie_indices = distances.argsort()[-6:-1][::-1]
    return df.iloc[movie_indices]

# ------------------------------
# SIDEBAR FAVORITES
# ------------------------------
st.sidebar.subheader("❤️ Favorites")

for fav in st.session_state.favorites:
    st.sidebar.write(fav)

# ------------------------------
# HOME PAGE
# ------------------------------
if st.session_state.page == "home":

    st.title("🎬 Movie Analytics Dashboard")

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("🎬 Total Movies", len(df))
    col2.metric("⭐ Avg Rating", round(df["Vote_Average"].mean(), 2))
    col3.metric("🔥 Max Popularity", round(df["Popularity"].max(), 2))

    # Filters
    st.sidebar.header("🔍 Filters")

    genre = st.sidebar.selectbox(
        "Select Genre",
        sorted(df["Genre"].dropna().unique())
    )

    search = st.sidebar.text_input("Search Movie")

    filtered_df = df[df["Genre"] == genre]

    if search:
        filtered_df = filtered_df[
            filtered_df["Title"].str.contains(search, case=False, na=False)
        ]

    # Movie Cards (Clickable)
    st.subheader("🎬 Movies")

    movies = filtered_df.head(12)
    cols = st.columns(4)

    for i, (_, row) in enumerate(movies.iterrows()):
        with cols[i % 4]:
            st.image(row["Poster_Url"], use_container_width=True)
            if st.button(row["Title"], key=i):
                st.session_state.selected_movie = row["Title"]
                st.session_state.page = "detail"
                st.rerun()

    # Chart
    st.subheader("📊 Popularity Distribution")

    fig, ax = plt.subplots()
    ax.hist(filtered_df["Popularity"], bins=10)
    st.pyplot(fig)

# ------------------------------
# DETAIL PAGE
# ------------------------------
elif st.session_state.page == "detail":

    movie_name = st.session_state.selected_movie
    movie = df[df["Title"] == movie_name].iloc[0]

    st.title(f"🎬 {movie['Title']}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(movie["Poster_Url"], use_container_width=True)

    with col2:
        st.write(f"⭐ Rating: {movie['Vote_Average']}")
        st.write(f"🔥 Popularity: {movie['Popularity']}")
        st.write(f"🎭 Genre: {movie['Genre']}")

        if "Overview" in df.columns:
            st.write("📖 Overview:")
            st.write(movie["Overview"])

        # Add to favorites
        if st.button("❤️ Add to Favorites"):
            st.session_state.favorites.append(movie_name)

    # Recommendations
    st.subheader("🎯 Similar Movies")

    results = recommend(movie_name)

    cols = st.columns(5)

    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i]:
            st.image(row["Poster_Url"], use_container_width=True)
            st.caption(row["Title"])

    # Back button
    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()