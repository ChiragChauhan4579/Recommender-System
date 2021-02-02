import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies_data = pd.read_csv('C:/Users/Chirag Chauhan/Desktop/ml-latest/movies.csv',usecols=['movieId','title','genres'],
                          dtype={'movieId':'int64'})
ratings_data = pd.read_csv('C:/Users/Chirag Chauhan/Desktop/ml-latest/ratings.csv',usecols=['userId','movieId','rating'],
                          dtype={'userId':'int64','movieId':'int64','rating':'float64'})
movies_data['title'] = movies_data['title'].str.split('(', 1).str[0].str.strip()

data = pd.merge(ratings_data,movies_data,on='movieId')

rating_count = data.dropna(axis=0,subset=['title'])
movie_rating_count = (rating_count.
                      groupby(by=['title'])['rating'].
                      count().
                      reset_index().
                      rename(columns={'rating':'totalRatings'})
                      [['title','totalRatings']]
                     )
pop_thres = 15000
movie_rating_count_ = movie_rating_count.query('totalRatings >= @pop_thres')
rating_with_totalRatings = pd.merge(rating_count,movie_rating_count,left_on='title',right_on='title',how='left')

rating_popular_movie = rating_with_totalRatings.query('totalRatings >= @pop_thres')

movie_features = rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)

movie_matrix_features = csr_matrix(movie_features.values)

model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
model_knn.fit(movie_matrix_features)

query_index = input('Enter the name of your movie: ')
#print(query_index)
for i in range(0,len(movie_rating_count_)):
    if movie_rating_count_.iloc[i]['title']== query_index:
         a = i
print(a)
distances,indices = model_knn.kneighbors(movie_features.iloc[a,:].values.reshape(1,-1),n_neighbors=6)
print(movie_features.iloc[a,:])

for i in range(0,len(distances.flatten())):
    if i == 0:
        print('Recommendation for {0}'.format(movie_features.index[a]))
    else:
        print('{0}:{1}'.format(i,movie_features.index[indices.flatten()[i]],distances.flatten()[i],movie_features.index[a]))