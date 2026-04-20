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
st.markdown("### 📊 Final Year Data Analytics Project")

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
# KPI Section
# ------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🎬 Total Movies", len(df))
col2.metric("⭐ Avg Rating", round(df["Vote_Average"].mean(), 2))
col3.metric("🔥 Max Popularity", round(df["Popularity"].max(), 2))

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("🔍 Filters")

genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + sorted(df["Genre"].dropna().unique().tolist())
)

language = st.sidebar.selectbox(
    "Select Language",
    sorted(df["Original_Language"].dropna().unique())
)

search_movie = st.sidebar.text_input("Search Movie")

# Apply filters
filtered_df = df.copy()

if genre != "All":
    filtered_df = filtered_df[filtered_df["Genre"] == genre]

if language:
    filtered_df = filtered_df[filtered_df["Original_Language"] == language]

if search_movie:
    filtered_df = filtered_df[
        filtered_df["Title"].str.contains(search_movie, case=False, na=False)
    ]

# ------------------------------
# Show Data
# ------------------------------
st.subheader(f"🎥 Movies in {genre} ({language})")
st.dataframe(filtered_df)

# Download option
st.download_button(
    "⬇ Download Data",
    filtered_df.to_csv(index=False),
    "filtered_movies.csv"
)

# ------------------------------
# Charts Section
# ------------------------------

# 1️⃣ Popularity Histogram
st.subheader("📊 Popularity Distribution")
fig, ax = plt.subplots()
ax.hist(filtered_df["Popularity"].dropna(), bins=15)
ax.set_xlabel("Popularity")
ax.set_ylabel("Count")
st.pyplot(fig)

# 2️⃣ Top 10 Movies
st.subheader("🏆 Top 10 Popular Movies")
top10 = filtered_df.sort_values("Popularity", ascending=False).head(10)
st.bar_chart(top10.set_index("Title")["Popularity"])

# 3️⃣ Genre Distribution
st.subheader("🎭 Genre Distribution")
genre_counts = df["Genre"].value_counts().head(10)

fig2, ax2 = plt.subplots()
ax2.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
st.pyplot(fig2)

# 4️⃣ Rating vs Popularity
st.subheader("📈 Rating vs Popularity")
st.scatter_chart(
    filtered_df,
    x="Vote_Average",
    y="Popularity"
)

# ------------------------------
# Insights Section
# ------------------------------
st.subheader("📌 Insights")

st.markdown("""
- High popularity movies generally have higher ratings  
- English movies dominate the dataset  
- Action & Drama genres appear most frequently  
- Popularity distribution is right-skewed  
""")