import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Movie Analytics Dashboard",
    layout="wide",
    page_icon="🎬"
)

# ------------------------------
# Custom UI Styling 🔥
# ------------------------------
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Analytics Dashboard")
st.markdown("### 🚀 AI-Powered Movie Recommendation System")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
    df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
    df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")

    df.dropna(inplace=True)

    return df

df = load_data()

# ------------------------------
# ML Model (Cosine Similarity)
# ------------------------------
@st.cache_data
def build_model(df):
    df["tags"] = df["Genre"] + " " + df["Overview"]

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return similarity

similarity = build_model(df)

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("🔍 Filters")

genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + sorted(df["Genre"].unique())
)

search = st.sidebar.text_input("Search Movie")

filtered_df = df.copy()

if genre != "All":
    filtered_df = filtered_df[filtered_df["Genre"] == genre]

if search:
    filtered_df = filtered_df[
        filtered_df["Title"].str.contains(search, case=False)
    ]

# ------------------------------
# KPI Section
# ------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🎬 Total Movies", len(filtered_df))
col2.metric("⭐ Avg Rating", round(filtered_df["Vote_Average"].mean(), 2))
col3.metric("🔥 Max Popularity", round(filtered_df["Popularity"].max(), 2))

# ------------------------------
# Show Movies with Posters 🎬
# ------------------------------
st.subheader("🎥 Movies")

if filtered_df.empty:
    st.warning("No data available")
else:
    cols = st.columns(5)

    for i, row in filtered_df.head(10).iterrows():
        with cols[i % 5]:
            st.image(row["Poster_Url"])
            st.caption(row["Title"])

# ------------------------------
# Charts
# ------------------------------
st.subheader("📊 Popularity Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df["Popularity"], bins=10)
st.pyplot(fig)

# ------------------------------
# ML Recommendation 🔥
# ------------------------------
st.subheader("🤖 AI Movie Recommendation")

movie_list = df["Title"].values
selected_movie = st.selectbox("Select a movie", movie_list)

def recommend(movie):
    index = df[df["Title"] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(df.iloc[i[0]])

    return recommended_movies

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    cols = st.columns(5)

    for i, movie in enumerate(recommendations):
        with cols[i]:
            st.image(movie["Poster_Url"])
            st.caption(movie["Title"])

# ------------------------------
# Insights
# ------------------------------
st.subheader("📌 Insights")

st.markdown("""
- AI-based recommendation using cosine similarity  
- Movies grouped using text features (genre + overview)  
- Visual dashboard for analysis  
""")