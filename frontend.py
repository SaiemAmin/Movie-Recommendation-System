import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import plotly.express as px

# ----------------- Page Configuration -----------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# ----------------- Custom CSS for Netflix-Inspired Theme & Enhanced Sidebar -----------------
st.markdown(
    """
    <style>
    /* Overall app styling */
    body {
        background-color: #000000;
        color: #E50914;
    }
    .stApp {
        background-color: #000000;
    }
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #E50914;
    }
    /* Movie cards styling */
    .card {
        background-color: #212121;
        padding: 10px;
        border-radius: 10px;
        margin: 10px;
        text-align: center;
    }
    /* Button styling */
    .stButton button {
        background-color: #E50914;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
    }
    /* Enhanced sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #141414, #000000) !important;
        color: #E50914;
        font-family: 'Helvetica Neue', sans-serif;
        padding: 20px;
    }
    /* Sidebar header text */
    [data-testid="stSidebar"] .css-1d391kg {
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- OS-Based Dimension Settings -----------------
if sys.platform == 'win32':
    MATPLOTLIB_FIGSIZE = (8, 6)
    MATPLOTLIB_DPI = 120
    PLOTLY_WIDTH = 600
    PLOTLY_HEIGHT = 400
else:
    MATPLOTLIB_FIGSIZE = (4, 3)
    MATPLOTLIB_DPI = 80
    PLOTLY_WIDTH = 400
    PLOTLY_HEIGHT = 250

plt.rcParams.update({'figure.figsize': MATPLOTLIB_FIGSIZE, 'figure.dpi': MATPLOTLIB_DPI})

# ----------------- Data Loading Function -----------------
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    required_cols = [
        'title', 'overview', 'genres', 'director',
        'poster_path', 'vote_average', 'vote_count', 'release_date'
    ]
    for col in required_cols:
        if col not in movies.columns:
            movies[col] = ''
    movies['combined_features'] = (
        movies['overview'].fillna('') + ' ' +
        movies['genres'].fillna('') + ' ' +
        movies['director'].fillna('')
    )
    movies = movies.drop_duplicates(subset='title').reset_index(drop=True)
    return movies

movies_data = load_data()

# ----------------- Compute TF-IDF and Cosine Similarity -----------------
@st.cache_data
def compute_tfidf_and_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

tfidf, tfidf_matrix, cosine_sim = compute_tfidf_and_similarity(movies_data)

# ----------------- Recommendation Function -----------------
def get_similar_movies(title, threshold=0.05, top_n=4):
    if title not in movies_data['title'].values:
        st.warning(f"'{title}' not found in the dataset.")
        return pd.DataFrame(columns=['title', 'poster_path', 'vote_average'])
    idx = movies_data[movies_data['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores 
                  if movies_data.iloc[score[0]]['title'] != title and score[1] >= threshold]
    seen = set()
    unique_sim_scores = []
    for i, score in sim_scores:
        movie_title = movies_data.iloc[i]['title']
        if movie_title not in seen:
            seen.add(movie_title)
            unique_sim_scores.append((i, score))
        if len(unique_sim_scores) >= top_n:
            break
    movie_indices = [i for i, score in unique_sim_scores]
    return movies_data.iloc[movie_indices][['title', 'poster_path', 'vote_average']]

# ----------------- Display Movies Grid Function (with embedded Details button) -----------------
def display_movies(movies):
    num_movies = len(movies)
    if num_movies == 0:
        st.write("No movies to display.")
        return
    cols = st.columns(min(num_movies, 5))
    for idx, movie in enumerate(movies.itertuples()):
        with cols[idx % 5]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if movie.poster_path:
                st.image(f"https://image.tmdb.org/t/p/w500{movie.poster_path}", width=150)
            else:
                st.image("https://via.placeholder.com/150", width=150)
            st.markdown(f"<h4 style='margin-bottom:2px;'>{movie.title}</h4>", unsafe_allow_html=True)
            st.write(f"🌟 Rating: {movie.vote_average}")
            if st.button("Details", key=f"details_{movie.title}_{idx}"):
                st.session_state.selected_movie = movie.title
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Plot Similarities using Plotly (Optimized Similarity Plot) -----------------
def plot_similarities(movie_title):
    idx = movies_data[movies_data['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores if movies_data.iloc[score[0]]['title'] != movie_title][:4]
    similar_titles = [movies_data.iloc[score[0]]['title'] for score in sim_scores]
    similarities = [score[1] for score in sim_scores]
    df = pd.DataFrame({"Title": similar_titles, "Similarity": similarities})
    fig = px.bar(df, x="Similarity", y="Title", orientation="h", title="Top Similar Movies",
                 height=PLOTLY_HEIGHT, width=PLOTLY_WIDTH, color_discrete_sequence = ["red"])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# ----------------- AI-Style Summary Functions -----------------
def generate_ai_summary(selected_title):
    summary = (
        f"The following movies were chosen because their overview, genres, and directorial style "
        f"exhibit strong thematic and stylistic similarities with '{selected_title}' in our collection. "
        f"This commonality suggests that the recommended movies share similar narrative and artistic qualities that might appeal to you."
    )
    return summary

def compute_avg_similarity(selected_title):
    idx = movies_data[movies_data['title'] == selected_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = [score for score in sim_scores if movies_data.iloc[score[0]]['title'] != selected_title]
    avg_sim = np.mean([score for i, score in sim_scores]) if sim_scores else 0
    return avg_sim

def display_summary_explanation(selected_title):
    summary = generate_ai_summary(selected_title)
    st.markdown(f"<p style='font-size:16px; line-height:1.5;'>{summary}</p>", unsafe_allow_html=True)

# ----------------- Insights Plots using Plotly -----------------
def plot_director_votes():
    director_votes = movies_data.groupby("director")["vote_count"].sum().reset_index()
    top_directors = director_votes.sort_values(by="vote_count", ascending=False).head(10)
    fig = px.bar(top_directors, x="vote_count", y="director", orientation="h", color_discrete_sequence= ["red"],
                 title="Top Directors by Vote Count", height=PLOTLY_HEIGHT, width=PLOTLY_WIDTH)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

def plot_yearly_votes():
    movies_data['release_year'] = pd.to_datetime(movies_data['release_date'], errors='coerce').dt.year  
    yearly_votes = movies_data.groupby("release_year")["vote_count"].sum().reset_index()
    fig = px.line(yearly_votes, x="release_year", y="vote_count", markers=True, color_discrete_sequence = ["red"],
                  title="Vote Counts by Year", height=PLOTLY_HEIGHT, width=PLOTLY_WIDTH)
    st.plotly_chart(fig, use_container_width=True)

def plot_top_movies():
    top_movies = movies_data.sort_values(by="vote_average", ascending=False).head(10)
    fig = px.bar(top_movies, x="title", y="vote_average", title="Top Movies by Average Vote",
                 height=PLOTLY_HEIGHT, width=PLOTLY_WIDTH, color_discrete_sequence=["red"])
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Display Movie Details (Netflix-Inspired Layout) -----------------
def display_movie_details(movie):
    st.markdown("## Movie Details")
    cols = st.columns([1, 2])
    with cols[0]:
        if movie.poster_path:
            st.image(f"https://image.tmdb.org/t/p/w500{movie.poster_path}", width=300)
        else:
            st.image("https://via.placeholder.com/300", width=300)
    with cols[1]:
        st.markdown(f"<h2>{movie.title}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Overview:</strong> {movie.overview}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Rating:</strong> {movie.vote_average}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Vote Count:</strong> {movie.vote_count}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Release Date:</strong> {movie.release_date}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Director:</strong> {movie.director}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Genres:</strong> {movie.genres}</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Recommended Movies")
    rec_df = get_similar_movies(movie.title)
    display_movies(rec_df)
    st.markdown("---")
    st.markdown("### Summary")
    display_summary_explanation(movie.title)
    st.markdown("---")
    st.markdown("### Similarity Visualization")
    plot_similarities(movie.title)
    if st.button("Back"):
        st.session_state.selected_movie = None

# ----------------- Handle Details View -----------------
if st.session_state.get("selected_movie"):
    selected_title = st.session_state.selected_movie
    movie_detail = movies_data[movies_data['title'] == selected_title]
    if not movie_detail.empty:
        movie_detail = movie_detail.iloc[0]
        display_movie_details(movie_detail)
else:
    # ----------------- Main Page Navigation -----------------
    selected_page = st.sidebar.radio("Navigation", ["Top Rated Movies", "Recommendations", "Insights"])

    if selected_page == "Top Rated Movies":
        st.title("🔥 Top Voted Movies")
        top_movies = movies_data.sort_values(by="vote_count", ascending=False).head(40)
        display_movies(top_movies)

    elif selected_page == "Recommendations":
        st.title("🎬 Content-Based Recommendations")
        query = st.text_input("Enter movie name:")
        if query:
            movie_titles = movies_data['title'].tolist()
            best_match_tuple = process.extractOne(query, movie_titles)
            best_match = best_match_tuple[0] if best_match_tuple else None
            if best_match:
                best_match_row = movies_data[movies_data['title'] == best_match]
                if not best_match_row.empty:
                    poster_url = best_match_row.iloc[0]['poster_path']
                    if poster_url:
                        st.sidebar.image("https://image.tmdb.org/t/p/w500" + poster_url, width=600)
                st.markdown("### Recommended Movies")
                rec_df = get_similar_movies(best_match)
                display_movies(rec_df)
                st.markdown("---")
                display_summary_explanation(best_match)
                st.markdown("---")
                st.markdown("### Similarity Visualization")
                plot_similarities(best_match)
        else:
            st.markdown("Please enter a movie name above.")

    elif selected_page == "Insights":
        st.title("📊 Insights")
        plot_director_votes()
        plot_yearly_votes()
        plot_top_movies()
