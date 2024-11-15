import streamlit as st
import difflib
import pickle
import requests

movies = pickle.load(open('movies_list.pkl' , 'rb'))
similarity = pickle.load(open('similarity.pkl' , 'rb'))
movies_titles = movies['title'].values

st.header("Movie Recommender System")

selected_movie = st.selectbox("Select movies from dropdown" , movies_titles)

def fetch_poster(movie_id):
     url = "https://api.themoviedb.org/3/movie/{}?api_key=407b23b1c23087553a5f4ea8d6a7e1f2&language=en-US".format(movie_id)
     data=requests.get(url)
     data=data.json()
     poster_path = data['poster_path']
     full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
     return full_path

def recommend(movie_name):
    find_close_matches = difflib.get_close_matches(movie_name , movies_titles)
    close_match = find_close_matches[0]
    index = movies[movies['title'] == close_match].index[0]
    distance = sorted(list(enumerate(similarity[index])), key = lambda x : x[1] , reverse = True)
    recommended_movies = []
    recommended_poster = []

    for i in distance[1:6]:
        movies_id=movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_poster.append(fetch_poster(movies_id))
    return recommended_movies, recommended_poster

if st.button("Show recommend"):
    
    movie_name , movie_poster = recommend(selected_movie)
    col1 , col2 , col3, col4 , col5 = st.columns(5)

    with col1:
        st.text(movie_name[0])
        st.image(movie_poster[0])
    with col2:
        st.text(movie_name[1])
        st.image(movie_poster[1])
    with col3:
        st.text(movie_name[2])
        st.image(movie_poster[2])
    with col4:
        st.text(movie_name[3])
        st.image(movie_poster[3])
    with col5:
        st.text(movie_name[4])
        st.image(movie_poster[4])