# Recommender-Systems-using-ML

**This example is based on collaborative filtering. An example of collaborative filtering method is when you buy a product through a website you would be getting recommendations of similar kind of products.<br />
<br />
The datasets movies and ratings can be downloaded from [here](http://grouplens.org/datasets/movielens/latest/)<br />
The libraries used here are numpy and pandas<br />
Importing datasets and checking their contents
```python
import numpy as np
import pandas as pd
movies_data = pd.read_csv('movies.csv',usecols=['movieId','title'])
ratings_data = pd.read_csv('ratings.csv',usecols=['userId','movieId','rating'])
```
