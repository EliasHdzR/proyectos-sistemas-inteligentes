from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

# Create our vectorizer
vectorizer = CountVectorizer()

# Let's fetch all the possible text data
newsgroups_data = fetch_20newsgroups()

print(newsgroups_data.data[130])