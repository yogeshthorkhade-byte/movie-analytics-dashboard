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
# NETFLIX STYLE UI
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
.movie-card {
    background-color: #1c1c1c;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Analytics Dashboard")
st.markdown("### 📊 Final Year Data Analytics Project")

# ------------------------------
# LOAD DATA (DEPLOYMENT SAFE)
# ------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "movies.csv",
            encoding="latin1",   # fixes encoding issue
            engine="python",
            on_bad_lines="skip"
        )

        df.columns = df.columns.str.strip()

        required = ["Title", "Genre", "Popularity", "Vote_Average", "Vote_Count"]

        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
        df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
        df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")

        df.dropna(inplace=True)

        return df

    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ Data failed to load. Check CSV file.")
    st.stop()

# ------------------------------
# KPI SECTION
# ------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🎬 Total Movies", len(df))
col2.metric("⭐ Avg Rating", round(df["Vote_Average"].mean(), 2))
col3.metric("🔥 Max Popularity", round(df["Popularity"].max(), 2))

# ------------------------------
# SIDEBAR FILTERS
# ------------------------------
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

# ------------------------------
# PREMIUM MOVIE LIST
# ------------------------------
st.subheader(f"🎥 Movies in {genre}")

if filtered_df.empty:
    st.warning("No movies found")
else:
    for _, row in filtered_df.head(10).iterrows():
        st.markdown(f"""
        🎬 **{row['Title']}**  
        ⭐ Rating: {row['Vote_Average']}  
        🔥 Popularity: {row['Popularity']}  
        """)
        st.markdown("---")

# ------------------------------
# DOWNLOAD BUTTON
# ------------------------------
st.download_button(
    "⬇ Download Data",
    filtered_df.to_csv(index=False),
    "movies.csv"
)

# ------------------------------
# POPULARITY GRAPH
# ------------------------------
st.subheader("📊 Popularity Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df["Popularity"], bins=10)
ax.set_xlabel("Popularity")
ax.set_ylabel("Count")
st.pyplot(fig)

# ------------------------------
# TOP 10 MOVIES
# ------------------------------
st.subheader("🏆 Top 10 Movies")

top10 = df.sort_values("Popularity", ascending=False).head(10)
st.bar_chart(top10.set_index("Title")["Popularity"])

# ------------------------------
# 🎬 POSTER CARDS (NETFLIX STYLE)
# ------------------------------
st.subheader("🎬 Featured Movies")

if "Poster_Url" in df.columns:

    movies = df.head(12)
    cols = st.columns(4)

    for i, (_, row) in enumerate(movies.iterrows()):
        with cols[i % 4]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            st.image(row["Poster_Url"], use_container_width=True)
            st.markdown(f"**{row['Title']}**")
            st.caption(f"⭐ {row['Vote_Average']}")
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("No posters available")

# ------------------------------
# 🤖 ML RECOMMENDATION SYSTEM
# ------------------------------
st.subheader("🤖 Smart Movie Recommendation")

try:
    df["combined"] = (
        df["Genre"].astype(str) + " " +
        df.get("Overview", "").astype(str)
    )

    cv = CountVectorizer(stop_words='english')
    matrix = cv.fit_transform(df["combined"])

    similarity = cosine_similarity(matrix)

    movie_list = df["Title"].values
    selected_movie = st.selectbox("Select a movie", movie_list)

    def recommend(movie):
        idx = df[df["Title"] == movie].index[0]
        distances = similarity[idx]
        movie_indices = distances.argsort()[-6:-1][::-1]
        return df.iloc[movie_indices]

    if st.button("Recommend Movies"):

        results = recommend(selected_movie)

        st.subheader("🎯 Recommended Movies")

        cols = st.columns(5)

        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i]:
                if "Poster_Url" in df.columns:
                    st.image(row["Poster_Url"], use_container_width=True)
                st.caption(row["Title"])

except:
    st.warning("Recommendation system unavailable")