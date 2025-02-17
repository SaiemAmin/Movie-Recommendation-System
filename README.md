# Movie Recommendation System

A content-based movie recommendation system that leverages data from the TMDB API (including over 40K movies and associated crew information) to deliver personalized movie suggestions. The system employs advanced data preprocessing techniques and state-of-the-art similarity measures to identify films that align with users' tastes. The application is built using Python, Streamlit, Plotly, and various data science libraries, and it features a modern, Netflix-inspired UI.

**Application**: https://movie-recommendation-system-rmpj3xxvdo5rjvbn8s8esm.streamlit.app/
---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Collection & Preprocessing](#data-collection--preprocessing)
- [Similarity Techniques](#similarity-techniques)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

---

## Overview

The Movie Recommendation System is designed to help users discover movies based on content similarity. By processing extensive movie and crew data from the TMDB API and applying advanced natural language processing techniques, the system identifies movies that share similar themes, genres, and directorial styles. Users can search for a movie and receive a list of recommended films along with visual insights and a detailed explanation of why those movies were chosen.

---

## Features

- **Content-Based Recommendations:**  
  Utilizes TF-IDF vectorization and cosine similarity to recommend movies that are textually similar based on overviews, genres, and director information.

  **Genre-Based Recommendation:**:
  Provides users a select-box of genres which recommends top 50 movies based on the genre they select.

- **Fuzzy Matching Search:**  
  Implements fuzzy string matching to handle imperfect user inputs, ensuring the best possible match is found even if there are typos.

- **Interactive and Modern UI:**  
  Built with Streamlit and styled with custom CSS to provide a sleek, Netflix-inspired dark theme with red accents. Interactive visualizations are provided using Plotly.

- **Multi-Page Layout:**  
  The application includes separate pages for Top Rated Movies, Content-Based Recommendations, and Data Insights.

- **Detailed Movie Views:**  
  Each movie card includes a details button that brings up a dedicated view with a larger poster, comprehensive details, recommended movies, a summary explanation, and similarity visualizations.

---

## Data Collection & Preprocessing

### Data Collection

- **TMDB API Integration:**  
  - **Movies API:** Retrieves detailed information for each movie, such as title, overview, genres (as IDs), poster path, ratings, vote counts, and release dates.
  - **Crew API:** Retrieves additional crew details (e.g., director information) for each movie.
  - **Merging Data:** The data from both API calls is merged into a single CSV file (`movies.csv`), resulting in a comprehensive dataset with over 40,000 movies.

### Data Preprocessing

- **Handling Missing Data:**  
  Ensured that all necessary columns (e.g., `title`, `overview`, `genres`, `director`, etc.) exist. If a column is missing, it is created with default empty values.

- **Feature Engineering:**  
  - **Mapping Genre IDs:** Genre IDs from TMDB are converted to human-readable genre names.
  - **Combined Features:** Key textual information—such as the movie overview, genres, and director—is concatenated into a single string (`combined_features`) for each movie. This combined text is used to compute similarities between movies.

- **Data Cleaning:**  
  Duplicates are removed to ensure each movie is unique, and the DataFrame is reset for consistent indexing.

---

## Similarity Techniques

- **TF-IDF Vectorization:**  
  The combined textual features are transformed into a TF-IDF matrix. This process quantifies the importance of each word in the context of the entire dataset, making it easier to compare the content of different movies.

- **Cosine Similarity:**  
  Cosine similarity is computed between the TF-IDF vectors of movies. This measure determines how similar two movies are based on the angle between their vector representations. A higher cosine similarity indicates more similar content.

- **Fuzzy Matching:**  
  To handle user queries that may contain typos or slight variations, fuzzy string matching (via the fuzzywuzzy library) is used to determine the best matching movie title from the dataset.

---

## Technologies Used

- **Python:** Core programming language.
- **Streamlit:** Web application framework for building interactive UI.
- **Pandas:** Data manipulation and analysis.
- **scikit-learn:** Machine learning library used for TF-IDF vectorization and cosine similarity.
- **Plotly:** Interactive data visualization.
- **FuzzyWuzzy:** Fuzzy string matching for search functionality.
- **Matplotlib:** Used for initial plotting (in some parts) and setting global plot parameters.
- **Custom CSS:** For styling and theming the UI (Netflix-inspired black and red theme).

---

## Usage

1. **Prepare the Data:**  
   Ensure that your `movies.csv` file (generated by merging TMDB API data) is placed in the project root.

2. **Install Dependencies:**  
   Use the provided `requirements.txt` file or install packages manually:
   ```bash
   pip install -r requirements.txt

   movie-recommender/
## Project Structure

├── app.py # Main Streamlit application script

├── movies.csv # CSV file containing merged movie and crew data from TMDB

├── requirements.txt # List of project dependencies

└── README.md # Project documentation



---

This README provides a structured and detailed description of your project for GitHub, covering all the key aspects—from what the project does to the methods used for preprocessing and computing similarities. Feel free to customize or expand on any sections as needed.

