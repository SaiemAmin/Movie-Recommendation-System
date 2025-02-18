import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
import plotly.express as px

# ----------------- Page Configuration -----------------
# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# ----------------- Custom CSS for Netflix-Inspired Theme & Enhanced Sidebar -----------------
st.markdown(
    """
    <style>
    /* Set the full background to pure black */
    .stApp {
        background-color: #000000 !important;
        color: #FFFFFF !important; /* White text for contrast */
    }

    /* Centered Movie Card Layout */
    .movie-card {
        display: flex;
        flex-direction: column;
        align-items: flex-start; /* Align items to the left */
        justify-content: center;
        text-align: left; /* Left-align text */
        width: 100%;
        padding: 10px; /* Add padding for better spacing */
    }

    /* Movie Title - Truncate Long Titles and Left-Align */
    .movie-title {
        display: block;
        width: 100%;  /* Ensure title takes full width of the card */
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 16px;
        font-weight: bold;
        margin: 8px 0 5px 0; /* Adjust margin for proper spacing */
        text-align: left; /* Left-align the title */
        color: white !important;
    }

    /* Movie Rating Styling */
    .movie-rating {
        color: #FFD700; /* Gold for contrast */
        font-size: 14px;
        margin: 5px 0; /* Adjust margin for proper spacing */
        text-align: left; /* Left-align the rating */
    }

    /* Recommend Button - Netflix Red */
    div.stButton > button {
        background-color: #E50914 !important;  /* Netflix Red */
        color: white !important;
        border-radius: 5px;
        border: none;
        padding: 8px 12px;
        font-size: 14px;
        font-weight: bold;
        cursor: pointer;
        display: block;
        margin: 10px 0; /* Adjust margin for proper spacing */
        width: 100%; /* Make button full width */
    }

    /* Hover effect */
    div.stButton > button:hover {
        background-color: #b20710 !important;  /* Darker red on hover */
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #141414 !important; /* Netflix Dark Gray */
        color: white !important;
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
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000, min_df = 5, max_df = 0.8)
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
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)

            # Display movie poster
            if movie.poster_path:
                st.image(f"https://image.tmdb.org/t/p/w500{movie.poster_path}", width=150)
            else:
                st.image("https://via.placeholder.com/150", width=150)

            # Properly Centered & Truncated Movie Title
            st.markdown(f"<div class='movie-title'>{movie.title}</div>", unsafe_allow_html=True)

            # Movie Rating (Gold for contrast)
            st.markdown(f"<p class='movie-rating'>🌟 <strong>Rating:</strong> {movie.vote_average}</p>", unsafe_allow_html=True)

            # Centered Red Recommend Button
            if st.button("Details", key=f"recommend_{movie.title}_{idx}"):
                st.session_state.selected_movie = movie.title
                st.query_params["dummy"] = str(np.random.randint(0, 100000))
                st.rerun()

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
                 title="Top Directors in the industry", height=PLOTLY_HEIGHT, width=PLOTLY_WIDTH)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

def plot_yearly_votes():
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

def plot_top_actors():
    top_actors = movies_data['actors'].str.split(",").explode().value_counts().head(10)
    fig = px.bar(top_actors, x=top_actors.index, y=top_actors.values, title="Most active Actors in the industry",
                 height=PLOTLY_HEIGHT, width=PLOTLY_WIDTH, color_discrete_sequence=["red"])
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def plot_regions():
    regions = movies_data['original_language'].value_counts().head(10)
    fig = px.bar(regions, x=regions.index, y=regions.values, title="Movies by Language",
                    height=PLOTLY_HEIGHT, width=PLOTLY_WIDTH, color_discrete_sequence=["red"])
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def plot_genres():
    genres = movies_data['genres'].str.split(",").explode().value_counts()
    fig = px.bar(genres, x=genres.index, y=genres.values, title="Most watched Genres",
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
        st.markdown(f"<p><strong>Release Date:</strong> {round(movie.release_year)}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Director:</strong> {movie.director}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Genres:</strong> {movie.genres}</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3 style='color: red;'>Recommended Movies</h3>", unsafe_allow_html=True)
    rec_df = get_similar_movies(movie.title)
    display_movies(rec_df)
    st.markdown("---")
    st.markdown("<h3 style='color: red;'>Summary</h3>", unsafe_allow_html=True)
    display_summary_explanation(movie.title)
    st.markdown("---")
    st.markdown("<h3 style='color: red;'>Similarity Visualization</h3>", unsafe_allow_html=True)
    plot_similarities(movie.title)
    if st.button("Back"):
        st.session_state.selected_movie = None
        # Restore the previous page if it exists
        if "prev_page" in st.session_state:
            st.session_state.selected_page = st.session_state.prev_page
        st.rerun()

# ----------------- Genre Recommendations Page -----------------
def genre_recommendations():
    st.title("🎬 Genre Recommendations")
    
    # Dropdown for genre selection
    genres = get_unique_genres(movies_data)
    selected_genre = st.selectbox("Select a Genre", genres)
    
    if selected_genre:
        # Remove rows with NA values in the genres column
        movies_data_cleaned = movies_data.dropna(subset=['genres'])
        
        # Filter movies by selected genre and sort by vote_average
        filtered_movies = movies_data_cleaned[movies_data_cleaned['genres'].str.contains(selected_genre, case=False)]
        top_movies = filtered_movies.sort_values(by='vote_average', ascending=False).head(50)
        
        if not top_movies.empty:
            st.write(f"**Top Voted Movies in '{selected_genre}' Genre:**")
            cols = st.columns(min(len(top_movies), 5))
            for idx, movie in enumerate(top_movies.itertuples()):
                with cols[idx % 5]:
                    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                    
                    # Display movie poster
                    if movie.poster_path:
                        st.image(f"https://image.tmdb.org/t/p/w500{movie.poster_path}", width=150)
                    else:
                        st.image("https://via.placeholder.com/150", width=150)
                    
                    # Truncate the title if it's too long
                    max_title_length = 20  # Adjust this value as needed
                    truncated_title = (movie.title[:max_title_length] + '...') if len(movie.title) > max_title_length else movie.title
                    
                    # Display truncated title
                    st.markdown(f"<div class='movie-title'>{truncated_title}</div>", unsafe_allow_html=True)
                    
                    # Display rating
                    st.markdown(f"<p class='movie-rating'>🌟 <strong>Rating:</strong> {movie.vote_average}</p>", unsafe_allow_html=True)
                    
                    if st.button("Details", key=f"details_{movie.title}_{idx}"):
                        st.session_state["prev_page"] = st.session_state.get("selected_page", "Top Rated Movies")
                        st.session_state["selected_movie"] = movie.title
                        st.rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write(f"No movies found in the '{selected_genre}' genre.")

@st.cache_data
def get_unique_genres(data):
    genres = data['genres'].str.split(',').explode().dropna().unique()
    genres = [genre for genre in genres if isinstance(genre, str)]
    return sorted(genres)

# ----------------- Handle Details View -----------------
if st.session_state.get("selected_movie"):
    selected_title = st.session_state.selected_movie
    movie_detail = movies_data[movies_data['title'] == selected_title]
    if not movie_detail.empty:
        movie_detail = movie_detail.iloc[0]
        display_movie_details(movie_detail)
else:
    # ----------------- Main Page Navigation -----------------
    selected_page = st.sidebar.radio("Navigation", ["Top Rated Movies", "Recommendations", "Genre Recommendations","Insights"],
                                     key = "selected_page")

    if selected_page == "Top Rated Movies":
        st.title("🔥 Top Rated Movies")
        top_movies = movies_data.sort_values(by="vote_count", ascending=False).head(25)
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
                    
                    st.markdown("## Movie Details")
                    cols = st.columns([1, 2])
                    with cols[0]:
                        if best_match_row.iloc[0]['poster_path']:
                            st.image(f"https://image.tmdb.org/t/p/w500{best_match_row.iloc[0]['poster_path']}", width=300)
                        else:
                            st.image("https://via.placeholder.com/300", width=300)
                    with cols[1]:
                        st.markdown(f"<h2>{best_match_row.iloc[0]['title']}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Overview:</strong> {best_match_row.iloc[0]['overview']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Rating:</strong> {best_match_row.iloc[0]['vote_average']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Vote Count:</strong> {best_match_row.iloc[0]['vote_count']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Release Date:</strong> {best_match_row.iloc[0]['release_date']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Director:</strong> {best_match_row.iloc[0]['director']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Genres:</strong> {best_match_row.iloc[0]['genres']}</p>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown("<h3 style='color: red;'>Recommended Movies</h3>", unsafe_allow_html=True)
                    rec_df = get_similar_movies(best_match)
                    display_movies(rec_df)
                    st.markdown("---")
                    st.markdown("<h3 style='color: red;'>Summary</h3>", unsafe_allow_html=True)
                    display_summary_explanation(best_match)
                    st.markdown("---")
                    st.markdown("<h3 style='color: red;'>Similarity Visualization</h3>", unsafe_allow_html=True)
                    plot_similarities(best_match)
        else:
            st.markdown("Please enter a movie name above.")
        
    elif selected_page == "Genre Recommendations":
        genre_recommendations()

    elif selected_page == "Insights":
        st.title("📊 Insights")
        plot_director_votes()
        plot_top_actors()
        plot_genres()
        plot_yearly_votes()
        plot_regions()
        plot_genres()
        plot_yearly_votes()
        plot_regions()
