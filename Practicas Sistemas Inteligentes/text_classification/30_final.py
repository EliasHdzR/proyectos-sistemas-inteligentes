from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()
feature_names = dataset['feature_names']
print("Feature names: {}\n".format(feature_names))

print("number of samples in the file (number of rows): ", dataset.data.shape[0])
print("number of features per row (columns): ", dataset.data.shape[1])

import pandas as pd
data_df = pd.DataFrame(dataset.data)

data_df.columns = ["MedInc", "HouseAge", "AveRooms",
                   "AveBedrms", "Population", "AveOccup",
                   "Latitude", "Longitude"]


print(data_df.head(5))
print(data_df[data_df['AveRooms']>100])

max_ave_rooms = 12
# column with index 2 corresponds to average number of rooms
shape = dataset.data.shape
cleansed_shape = dataset.data[dataset.data[:,2] <= max_ave_rooms].shape
print(shape, cleansed_shape)
n_outliers = shape[0]-cleansed_shape[0]
print(f"Number of outliers, more than {max_ave_rooms} bedrooms: {n_outliers}")

x = dataset.data[:,2] <= max_ave_rooms  # Boolean array
data = dataset.data[x]
targets = dataset.target[x]
data.shape, targets.shape

from sklearn.model_selection import train_test_split
data_sets = train_test_split(data,
                             targets,
                             test_size=0.30,
                             random_state=42)

data_train, data_test, targets_train, targets_test = data_sets

from sklearn.model_selection import train_test_split
data_sets = train_test_split(dataset.data,
                             dataset.target,
                             test_size=0.30,
                             random_state=42)

data_train2, data_test2, targets_train2, targets_test2 = data_sets

clf = MLPRegressor(solver='lbfgs',        #  ‘lbfgs’, ‘sgd’, ‘adam’ (default)
                   alpha=1e-5,            # used for regularization, ovoiding overfitting by penalizing large magnitudes
                   hidden_layer_sizes=(10, 2),
                   activation='logistic', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’ (default)
                   max_iter=10000,
                   random_state=42)
clf.fit(data_train, targets_train)
print(clf.score(data_train, targets_train))
print(clf.score(data_test, targets_test))

clf = MLPRegressor(solver='lbfgs',        #  ‘lbfgs’, ‘sgd’, ‘adam’ (default)
                   alpha=1e-5,            # used for regularization, ovoiding overfitting by penalizing large magnitudes
                   hidden_layer_sizes=(10, 2),
                   activation='logistic', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’ (default)
                   max_iter=10000,
                   random_state=42)
clf.fit(data_train2, targets_train2)
print(clf.score(data_train2, targets_train2))
print(clf.score(data_test2, targets_test2))

from sklearn import preprocessing

data_scaled = preprocessing.scale(data)

from sklearn.preprocessing import PolynomialFeatures
import numpy as np

pft = PolynomialFeatures(degree=2)
data_poly = pft.fit_transform(data_scaled)
data_poly

from sklearn.model_selection import train_test_split
data_sets = train_test_split(data_poly,
                             targets,
                             test_size=0.30,
                             random_state=42)

data_train, data_test, targets_train, targets_test = data_sets

clf = MLPRegressor(solver='lbfgs',   #  ‘lbfgs’, ‘sgd’, ‘adam’ (default)
                   alpha=1e-5,     # used for regularization, ovoiding overfitting by penalizing large magnitudes
                   hidden_layer_sizes=(5, 2),
                   activation='relu', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’ (default)
                   max_iter=10000,
                   early_stopping=True,
                   random_state=42)
clf.fit(data_train, targets_train)
print(clf.score(data_train, targets_train))
print(clf.score(data_test, targets_test))