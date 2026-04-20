import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Movie Analytics Dashboard",
    layout="wide",
    page_icon="🎬"
)

st.title("🎬 Movie Analytics Dashboard")
st.markdown("### 🚀 Final Year Data Analytics Project")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "movies.csv",
        sep=",",
        engine="python",
        encoding="utf-8",
        quotechar='"'
    )

    # Clean numeric columns
    df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
    df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
    df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")

    return df

df = load_data()

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("🔍 Filters")

genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + sorted(df["Genre"].dropna().unique())
)

language = st.sidebar.selectbox(
    "Select Language",
    ["All"] + sorted(df["Original_Language"].dropna().unique())
)

search_movie = st.sidebar.text_input("Search Movie")

# ------------------------------
# Filtering Logic
# ------------------------------
filtered_df = df.copy()

if genre != "All":
    filtered_df = filtered_df[filtered_df["Genre"] == genre]

if language != "All":
    filtered_df = filtered_df[filtered_df["Original_Language"] == language]

if search_movie:
    filtered_df = filtered_df[
        filtered_df["Title"].str.contains(search_movie, case=False, na=False)
    ]

# ------------------------------
# KPI Section
# ------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🎬 Total Movies", len(filtered_df))
col2.metric("⭐ Avg Rating", round(filtered_df["Vote_Average"].mean(), 2))
col3.metric("🔥 Max Popularity", round(filtered_df["Popularity"].max(), 2))

# ------------------------------
# Show Data
# ------------------------------
st.subheader(f"🎥 Movies Data")

if filtered_df.empty:
    st.warning("⚠ No data available for selected filters")
else:
    st.dataframe(filtered_df)

# Download button
st.download_button(
    "⬇ Download Data",
    filtered_df.to_csv(index=False),
    "filtered_movies.csv"
)

# ------------------------------
# Charts Section
# ------------------------------
if not filtered_df.empty:

    # Popularity Histogram
    st.subheader("📊 Popularity Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["Popularity"].dropna(), bins=15)
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Top 10 Movies
    st.subheader("🏆 Top 10 Popular Movies")
    top10 = filtered_df.sort_values("Popularity", ascending=False).head(10)
    st.bar_chart(top10.set_index("Title")["Popularity"])

    # Rating vs Popularity
    st.subheader("📈 Rating vs Popularity")
    st.scatter_chart(filtered_df, x="Vote_Average", y="Popularity")

    # Language Distribution
    st.subheader("🌍 Language Distribution")
    lang_counts = df["Original_Language"].value_counts().head(10)

    fig2, ax2 = plt.subplots()
    ax2.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%')
    st.pyplot(fig2)

# ------------------------------
# Recommendation System 🔥
# ------------------------------
st.subheader("🤖 Movie Recommendation System")

movie_list = df["Title"].dropna().unique()

selected_movie = st.selectbox("Choose a movie", movie_list)

if selected_movie:

    # Recommend based on same genre
    movie_genre = df[df["Title"] == selected_movie]["Genre"].values[0]

    recommendations = df[df["Genre"] == movie_genre] \
        .sort_values("Popularity", ascending=False) \
        .head(5)

    st.write("### 🎯 Recommended Movies:")
    st.dataframe(recommendations[["Title", "Popularity", "Vote_Average"]])

# ------------------------------
# Insights Section
# ------------------------------
st.subheader("📌 Insights")

st.markdown("""
- 🎯 High popularity movies tend to have higher ratings  
- 🌍 English dominates but multilingual content is rising  
- 🎬 Action & Drama genres are most frequent  
- 📊 Popularity shows a right-skewed distribution  
- 🤖 Recommendation system suggests similar genre movies  
""")