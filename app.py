import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Movie Analytics Dashboard", layout="wide")

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

    # Fix numeric
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
# SIDEBAR
# ------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Dashboard", "Recommendations"]
)

# ------------------------------
# HOME
# ------------------------------
if page == "Home":

    st.title("Movie Analytics & Recommendation System")

    search = st.text_input("Search Movie")

    if search:
        results = df[df["Title"].str.contains(search, case=False, na=False)]

        st.write(f"Results: {len(results)}")
        st.dataframe(results[["Title", "Genre", "Vote_Average"]])

# ------------------------------
# DASHBOARD
# ------------------------------
elif page == "Dashboard":

    st.title("Movie Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", len(df))
    col2.metric("Average Rating", round(df["Vote_Average"].mean(), 2))
    col3.metric("Max Popularity", round(df["Popularity"].max(), 2))

    genre = st.selectbox("Select Genre", df["Genre"].unique())
    filtered_df = df[df["Genre"] == genre]

    # Histogram
    st.subheader("Popularity Distribution")

    fig, ax = plt.subplots()
    ax.hist(filtered_df["Popularity"], bins=10)
    st.pyplot(fig)

    # Bar chart
    st.subheader("Top Movies")

    top10 = filtered_df.sort_values("Popularity", ascending=False).head(10)
    st.bar_chart(top10.set_index("Title")["Popularity"])

# ------------------------------
# RECOMMENDATION
# ------------------------------
elif page == "Recommendations":

    st.title("Movie Recommendation System")

    df["combined"] = (df["Genre"] + " " + df["Overview"]).fillna("")

    cv = CountVectorizer(stop_words='english')
    matrix = cv.fit_transform(df["combined"])

    similarity = cosine_similarity(matrix)

    movie_list = df["Title"].values
    selected_movie = st.selectbox("Select Movie", movie_list)

    def recommend(movie):
        idx = df[df["Title"] == movie].index[0]
        distances = similarity[idx]

        movie_indices = distances.argsort()[-6:-1][::-1]
        return df.iloc[movie_indices]

    if st.button("Recommend"):

        results = recommend(selected_movie)

        st.subheader("Recommended Movies")

        for _, row in results.iterrows():
            st.write(row["Title"])