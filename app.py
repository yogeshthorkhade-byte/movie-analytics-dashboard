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
# CUSTOM UI
# ------------------------------
st.markdown("""
<style>
h1 { color: #E50914; }
.stButton>button { background-color: #E50914; color: white; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv", encoding="latin1", engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
st.sidebar.title("🎬 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Dashboard", "🤖 Recommendations"]
)

# ------------------------------
# HOME PAGE
# ------------------------------
if page == "🏠 Home":

    st.title("🎬 Movie Analytics & Recommendation System")

    st.markdown("""
    ### 🎓 Final Year Project

    This system provides:
    - 📊 Movie Data Analysis  
    - 🎬 Interactive Dashboard  
    - 🤖 AI-based Recommendation System  
    """)

    # 🔍 Enhanced Search
    st.subheader("🔍 Search Movie")

    search = st.text_input("Type movie name")

    if search:
        results = df[df["Title"].str.contains(search, case=False, na=False)]

        if results.empty:
            st.warning("No movies found")
        else:
            cols = st.columns(4)

            for i, (_, row) in enumerate(results.head(8).iterrows()):
                with cols[i % 4]:
                    st.image(row["Poster_Url"], use_container_width=True)
                    st.caption(row["Title"])

# ------------------------------
# DASHBOARD PAGE
# ------------------------------
elif page == "📊 Dashboard":

    st.title("📊 Movie Analytics Dashboard")

    # KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", len(df))
    col2.metric("Average Rating", round(df["Vote_Average"].mean(), 2))
    col3.metric("Max Popularity", round(df["Popularity"].max(), 2))

    # Filters
    genre = st.selectbox("Select Genre", df["Genre"].unique())

    filtered_df = df[df["Genre"] == genre]

    # Charts
    st.subheader("📈 Popularity Distribution")

    fig, ax = plt.subplots()
    ax.hist(filtered_df["Popularity"], bins=10)
    st.pyplot(fig)

    st.subheader("🏆 Top Movies")

    top10 = filtered_df.sort_values("Popularity", ascending=False).head(10)
    st.bar_chart(top10.set_index("Title")["Popularity"])

    st.subheader("📊 Rating vs Popularity")
    st.scatter_chart(filtered_df, x="Vote_Average", y="Popularity")

# ------------------------------
# RECOMMENDATION PAGE
# ------------------------------
elif page == "🤖 Recommendations":

    st.title("🤖 Movie Recommendation System")

    # ML model
    df["combined"] = df["Genre"].astype(str) + " " + df.get("Overview", "").astype(str)

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

    if st.button("Recommend"):

        results = recommend(selected_movie)

        st.subheader("🎯 Recommended Movies")

        cols = st.columns(5)

        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i]:
                st.image(row["Poster_Url"], use_container_width=True)
                st.caption(row["Title"])