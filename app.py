import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Movie Analytics Dashboard",
    layout="wide",
    page_icon="🎬"
)

st.title("🎬 Movie Analytics Dashboard")
st.markdown("### 📊 Final Year Data Analytics Project")

# ------------------------------
# Load Dataset (SAFE VERSION)
# ------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "movies.csv",
            sep=",",
            engine="python",
            encoding="utf-8",
            quotechar='"',
            on_bad_lines='skip'
        )

        df.columns = df.columns.str.strip()

        # Required columns
        required = ["Title", "Genre", "Popularity", "Vote_Average", "Vote_Count"]

        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        # Convert numeric safely
        df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
        df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
        df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")

        df.dropna(inplace=True)

        return df

    except Exception as e:
        return pd.DataFrame()

df = load_data()

# STOP if data fails
if df.empty:
    st.error("❌ Data failed to load. Check CSV format.")
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

# Filter
filtered_df = df[df["Genre"] == genre]

if search:
    filtered_df = filtered_df[
        filtered_df["Title"].str.contains(search, case=False, na=False)
    ]

# ------------------------------
# DISPLAY DATA
# ------------------------------
st.subheader(f"🎥 Movies in {genre}")

if filtered_df.empty:
    st.warning("No movies found")
else:
    st.dataframe(filtered_df.head(50))

# Download
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
# 🎬 ML RECOMMENDATION SYSTEM
# ------------------------------
st.subheader("🤖 AI Movie Recommendation")

try:
    df["combined"] = df["Genre"].astype(str)

    cv = CountVectorizer()
    matrix = cv.fit_transform(df["combined"])

    similarity = cosine_similarity(matrix)

    movie_list = df["Title"].values

    selected_movie = st.selectbox("Choose a movie", movie_list)

    def recommend(movie):
        idx = df[df["Title"] == movie].index[0]
        distances = list(enumerate(similarity[idx]))
        movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

        return [df.iloc[i[0]].Title for i in movies]

    if st.button("Recommend"):
        results = recommend(selected_movie)

        for m in results:
            st.write("👉", m)

except:
    st.warning("Recommendation system unavailable")

# ------------------------------
# POSTERS (OPTIONAL SAFE)
# ------------------------------
st.subheader("🎬 Movie Posters")

if "Poster_Url" in df.columns:
    sample = df.head(6)

    cols = st.columns(3)

    for i, (_, row) in enumerate(sample.iterrows()):
        with cols[i % 3]:
            st.image(row["Poster_Url"], width=200)
            st.caption(row["Title"])
else:
    st.info("Poster data not available")