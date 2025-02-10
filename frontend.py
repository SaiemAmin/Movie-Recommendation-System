import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Initialize session state for selected movie details
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# --- DATA LOADING ---
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    
    # Ensure required columns exist.
    required_cols = [
        'title', 'overview', 'genres', 'director', 
        'poster_path', 'vote_average', 'vote_count', 'release_date'
    ]
    for col in required_cols:
        if col not in movies.columns:
            movies[col] = ''
    
    # Combine text features for content-based filtering.
    movies['combined_features'] = (
        movies['overview'].fillna('') + ' ' +
        movies['genres'].fillna('') + ' ' +
        movies['director'].fillna('')
    )
    
    # Drop duplicate movies (based on title) and reset the index.
    movies = movies.drop_duplicates(subset='title').reset_index(drop=True)
    return movies

movies_data = load_data()

# --- COMPUTE COSINE SIMILARITY ---
@st.cache_data
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_similarity(movies_data)

# --- RECOMMENDATION FUNCTION ---
def get_similar_movies(title, threshold=0.05, top_n=4):
    """
    Returns up to top_n movies similar to the given title using cosine similarity.
    Excludes the queried movie and ensures unique recommendations.
    """
    if title not in movies_data['title'].values:
        st.warning(f"'{title}' not found in the dataset.")
        return pd.DataFrame(columns=['title', 'poster_path', 'vote_average'])
    
    idx = movies_data[movies_data['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the selected movie and any with similarity below the threshold.
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

# --- DISPLAY MOVIES FUNCTION ---
def display_movies(movies):
    """
    Displays a grid of movies (up to 5 per row) with poster, title, rating,
    and a Details button. Clicking Details saves the movie title in session state.
    """
    num_movies = len(movies)
    if num_movies == 0:
        st.write("No movies to display.")
        return
    
    cols = st.columns(min(num_movies, 5))
    for idx, movie in enumerate(movies.itertuples()):
        with cols[idx % 5]:
            if movie.poster_path:
                st.image(f"https://image.tmdb.org/t/p/w500{movie.poster_path}", width=150)
            else:
                st.image("https://via.placeholder.com/150", width=150)
            st.markdown(f"<h4 style='margin-bottom:5px;'>{movie.title}</h4>", unsafe_allow_html=True)
            st.write(f"🌟 Rating: {movie.vote_average}")
            if st.button("Details", key=f"details_{movie.title}_{idx}"):
                st.session_state.selected_movie = movie.title

# --- PLOT SIMILAR MOVIES (for details view) ---
def plot_similarities(movie_title):
    """
    For a given movie, plots a horizontal bar chart of the top similar movies
    (excluding the selected movie).
    """
    idx = movies_data[movies_data['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Exclude the selected movie and take the top 4 similar movies.
    sim_scores = [score for score in sim_scores if movies_data.iloc[score[0]]['title'] != movie_title][:4]
    similar_titles = [movies_data.iloc[score[0]]['title'] for score in sim_scores]
    similarities = [score[1] for score in sim_scores]
    
    fig, ax = plt.subplots()
    ax.barh(similar_titles, similarities, color="skyblue")
    ax.set_xlabel("Similarity Score")
    ax.set_title("Top Similar Movies")
    st.pyplot(fig)

# --- DISPLAY MOVIE DETAILS ---
def display_movie_details(movie):
    """
    Displays detailed information for a given movie and shows a plot
    of similar movies.
    """
    st.markdown("### Movie Details")
    if movie.poster_path:
        st.image(f"https://image.tmdb.org/t/p/w500{movie.poster_path}", width=200)
    else:
        st.image("https://via.placeholder.com/200", width=200)
    st.markdown(f"**Title:** {movie.title}")
    st.markdown(f"**Overview:** {movie.overview}")
    st.markdown(f"**Rating:** {movie.vote_average}")
    st.markdown(f"**Vote Count:** {movie.vote_count}")
    st.markdown(f"**Release Date:** {movie.release_date}")
    st.markdown(f"**Director:** {movie.director}")
    st.markdown(f"**Genres:** {movie.genres}")
    
    st.markdown("#### Similar Movies")
    plot_similarities(movie.title)

# --- MAIN APP LAYOUT ---

st.title("🎬 Movie Recommendation System")

# Sidebar: Movie Search for Recommendations
st.sidebar.header("Movie Search")
search_query = st.sidebar.text_input("Search for a movie:", "")
if search_query:
    movie_titles = movies_data['title'].tolist()
    # Use fuzzy matching to provide suggestions.
    matched = process.extract(search_query, movie_titles, limit=10)
    matched_titles = [match[0] for match in matched]
    selected_movie = st.sidebar.selectbox("Select a movie:", matched_titles)
    if selected_movie:
        movie_row = movies_data[movies_data['title'] == selected_movie]
        if not movie_row.empty and movie_row['poster_path'].values[0]:
            st.sidebar.image(f"https://image.tmdb.org/t/p/w500{movie_row['poster_path'].values[0]}")
        st.markdown(f"## Movies similar to **{selected_movie}**")
        recommended_movies = get_similar_movies(selected_movie)
        display_movies(recommended_movies)

# Navigation Tabs on the Sidebar
selected_tab = st.sidebar.radio("Navigate", ["Search", "Popular Movies", "Insights"])

if selected_tab == "Search":
    st.subheader("🔍 Search for a movie")
    query = st.text_input("Enter movie name:", "")
    if query:
        movie_titles = movies_data['title'].tolist()
        matched = process.extract(query, movie_titles, limit=10)
        matched_titles = [match[0] for match in matched]
        filtered_movies = movies_data[movies_data['title'].isin(matched_titles)]
    else:
        filtered_movies = movies_data.head(10)
    display_movies(filtered_movies)

elif selected_tab == "Popular Movies":
    st.subheader("🔥 Top Rated Movies")
    top_movies = movies_data.sort_values(by="vote_average", ascending=False).head(10)
    display_movies(top_movies)

elif selected_tab == "Insights":
    st.subheader("📊 Insights")
    def plot_director_votes():
        st.subheader("🎬 Top 10 Directors by Vote Count")
        director_votes = movies_data.groupby("director")["vote_count"].sum().reset_index()
        top_directors = director_votes.sort_values(by="vote_count", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(top_directors["director"], top_directors["vote_count"], color="skyblue")
        ax.set_xlabel("Vote Count")
        ax.set_ylabel("Director")
        ax.set_title("Top Directors by Vote Count")
        plt.gca().invert_yaxis()
        st.pyplot(fig)
    
    def plot_yearly_votes():
        st.subheader("📈 Vote Counts by Year")
        movies_data['release_year'] = pd.to_datetime(movies_data['release_date'], errors='coerce').dt.year
        yearly_votes = movies_data.groupby("release_year")["vote_count"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(yearly_votes['release_year'], yearly_votes['vote_count'], marker='o', linestyle='-', color="skyblue")
        ax.set_title("Vote Counts by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Vote Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    def plot_top_movies():
        st.subheader("🎬 Top 10 Movies by Average Vote")
        top_movies = movies_data.sort_values(by="vote_average", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.bar(top_movies["title"], top_movies["vote_average"], color="coral")
        ax.set_title("Average Vote for Top Movies")
        ax.set_xlabel("Movie Title")
        ax.set_ylabel("Average Vote")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    
    plot_director_votes()
    plot_yearly_votes()
    plot_top_movies()

# --- SHOW MOVIE DETAILS IF SELECTED ---
if st.session_state.selected_movie is not None:
    selected_title = st.session_state.selected_movie
    movie_detail = movies_data[movies_data['title'] == selected_title]
    if not movie_detail.empty:
        movie_detail = movie_detail.iloc[0]
        st.markdown("---")
        display_movie_details(movie_detail)
        if st.button("Close Details"):
            st.session_state.selected_movie = None
