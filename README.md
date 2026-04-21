# 🎬 Movie Analytics & Recommendation System

An intelligent, data-driven web application that provides movie insights and personalized recommendations using Machine Learning. Built with Streamlit, this project simulates a real-world OTT analytics and recommendation platform.

---

## 🚀 Features

* 🔍 **Smart Movie Search**
  Search movies dynamically with filtering options.

* 🎯 **AI Recommendation System**
  Content-based filtering using cosine similarity.

* 📊 **Analytics Dashboard**
  KPI metrics like total movies, average rating, and popularity.

* 📈 **Trend Analysis**
  Visual insights on movie popularity and ratings.

* 🎬 **Genre-based Exploration**
  Filter and analyze movies by genre.

* ⭐ **Top Movies Visualization**
  Identify trending and top-rated movies.

---

## 🧠 Tech Stack

* **Frontend & App Framework:** Streamlit
* **Backend:** Python
* **Libraries:**

  * Pandas
  * NumPy
  * Matplotlib
  * Scikit-learn

---

## 📂 Project Structure

```
movie-analytics-dashboard/
│
├── app.py              # Main Streamlit app
├── movies.csv          # Dataset
├── requirements.txt    # Dependencies
├── cap_DS.ipynb        # Data analysis & model development
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yogeshthorkhade-byte/movie-analytics-dashboard.git
cd movie-analytics-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## 🤖 How Recommendation Works

* Combines **Genre + Overview** into a single feature
* Converts text into vectors using **CountVectorizer**
* Computes similarity using **Cosine Similarity**
* Recommends movies based on similarity scores

---

## 📊 Dataset

* Contains movie details such as:

  * Title
  * Genre
  * Rating
  * Popularity
  * Overview
  * Language

---

## 🎯 Project Objective

To build an interactive system that:

* Analyzes movie data
* Provides insights through visualization
* Recommends movies using machine learning

---

## 💡 Key Highlights

* End-to-end data analytics project
* Machine learning integration
* Interactive UI with Streamlit
* Real-world application use case

---

## 🎓 Future Improvements

* Hybrid recommendation system
* User login & watchlist
* Real-time API integration (TMDB)
* Netflix-style UI enhancements

---

## 👨‍💻 Author

**Yogesh Thorkhade**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
