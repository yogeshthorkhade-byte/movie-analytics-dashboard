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

# ------------------------------
# Load Dataset
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

    # Convert numeric columns properly
    df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
    df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
    df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")

    return df


df = load_data()

# ------------------------------
# KPI Section
# ------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Movies", len(df))
col2.metric("Average Rating", round(df["Vote_Average"].mean(), 2))
col3.metric("Max Popularity", round(df["Popularity"].max(), 2))

# ------------------------------
# Show Raw Dataset
# ------------------------------
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)

# ------------------------------
# Sidebar Filter
# ------------------------------
st.sidebar.header("Filter Options")

genre = st.sidebar.selectbox(
    "Select Genre",
    sorted(df["Genre"].dropna().unique())
)

search_movie = st.sidebar.text_input("Search Movie")

filtered_df = df[df["Genre"] == genre]

# ------------------------------
# Search Results
# ------------------------------
if search_movie:
    st.subheader("Search Results")
    search_results = df[df["Title"].str.contains(search_movie, case=False, na=False)]
    st.dataframe(search_results)

# ------------------------------
# Display Filtered Data
# ------------------------------
st.subheader(f"Movies in {genre}")
st.dataframe(filtered_df)

st.download_button(
    label="Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_movies.csv",
    mime="text/csv"
)

# ------------------------------
# Popularity Distribution
# ------------------------------
st.subheader("Popularity Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df["Popularity"].dropna(), bins=10)
ax.set_xlabel("Popularity")
ax.set_ylabel("Count")
st.pyplot(fig)

# ------------------------------
# Top 10 Most Popular Movies (Improved Chart)
# ------------------------------
st.subheader("Top 10 Most Popular Movies")

top10 = df.sort_values("Popularity", ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(top10["Title"], top10["Popularity"])
ax2.set_xlabel("Popularity")
ax2.set_title("Top 10 Movies by Popularity")
ax2.invert_yaxis()

st.pyplot(fig2)