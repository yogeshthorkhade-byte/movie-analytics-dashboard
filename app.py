import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Movie Analytics System", layout="wide")

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

    # Numeric fix
    df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
    df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
    df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")

    # Text fix
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
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dashboard", "🤖 Recommendations"])

# ------------------------------
# 🏠 HOME PAGE
# ------------------------------
if page == "🏠 Home":

    st.title("🎬 Movie Analytics & Recommendation System")

    st.markdown("### 🔍 Smart Movie Search")

    search = st.text_input("Search movies")

    if search:
        results = df[df["Title"].str.contains(search, case=False, na=False)]

        if results.empty:
            st.warning("No movies found")
        else:
            st.write(f"Found {len(results)} results")

            cols = st.columns(4)

            for i, (_, row) in enumerate(results.head(12).iterrows()):
                with cols[i % 4]:
                    st.image(row["Poster_Url"], use_container_width=True)
                    st.caption(f"{row['Title']} ⭐ {row['Vote_Average']}")

    # 🔥 Trending Section
    st.subheader("🔥 Trending Movies")

    trending = df.sort_values("Popularity", ascending=False).head(8)

    cols = st.columns(4)

    for i, (_, row) in enumerate(trending.iterrows()):
        with cols[i % 4]:
            st.image(row["Poster_Url"], use_container_width=True)
            st.caption(row["Title"])

# ------------------------------
# 📊 DASHBOARD PAGE
# ------------------------------
elif page == "📊 Dashboard":

    st.title("📊 Advanced Movie Analytics Dashboard")

    # KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", len(df))
    col2.metric("Avg Rating", round(df["Vote_Average"].mean(), 2))
    col3.metric("Max Popularity", round(df["Popularity"].max(), 2))

    # Filters
    st.subheader("🎯 Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        genre = st.selectbox("Genre", df["Genre"].unique())

    with col2:
        min_rating = st.slider("Min Rating", 0.0, 10.0, 5.0)

    with col3:
        min_pop = st.slider("Min Popularity", 0.0, float(df["Popularity"].max()), 10.0)

    filtered_df = df[
        (df["Genre"] == genre) &
        (df["Vote_Average"] >= min_rating) &
        (df["Popularity"] >= min_pop)
    ]

    st.write(f"Filtered Movies: {len(filtered_df)}")

    # Charts
    st.subheader("📈 Popularity Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["Popularity"], bins=15)
    st.pyplot(fig)

    st.subheader("🏆 Top Movies")
    top10 = filtered_df.sort_values("Popularity", ascending=False).head(10)
    st.bar_chart(top10.set_index("Title")["Popularity"])

    st.subheader("📊 Rating vs Popularity")
    st.scatter_chart(filtered_df, x="Vote_Average", y="Popularity")

    # 🔥 Top Rated Section
    st.subheader("⭐ Top Rated Movies")
    top_rated = df.sort_values("Vote_Average", ascending=False).head(10)
    st.dataframe(top_rated[["Title", "Vote_Average"]])

# ------------------------------
# 🤖 RECOMMENDATION PAGE
# ------------------------------
elif page == "🤖 Recommendations":

    st.title("🤖 AI Movie Recommendation System")

    # Prepare ML
    df["combined"] = (df["Genre"] + " " + df["Overview"]).fillna("")

    cv = CountVectorizer(stop_words='english')
    matrix = cv.fit_transform(df["combined"])

    similarity = cosine_similarity(matrix)

    movie_list = df["Title"].values
    selected_movie = st.selectbox("Select a movie", movie_list)

    def recommend(movie):
        idx = df[df["Title"] == movie].index[0]
        distances = similarity[idx]

        movie_indices = distances.argsort()[-6:-1][::-1]

        results = []
        for i in movie_indices:
            results.append((df.iloc[i], distances[i]))

        return results

    if st.button("Recommend Movies"):

        recommendations = recommend(selected_movie)

        st.subheader("🎯 Recommended Movies")

        cols = st.columns(5)

        for i, (row, score) in enumerate(recommendations):
            with cols[i]:
                st.image(row["Poster_Url"], use_container_width=True)
                st.caption(f"{row['Title']}")
                st.caption(f"Similarity: {round(score*100, 2)}%")