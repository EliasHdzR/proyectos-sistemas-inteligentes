from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()
from sklearn.neural_network import MLPRegressor


feature_names = dataset['feature_names']
print("Feature names: {}\n".format(feature_names))

print("number of samples in the file (number of rows): ", dataset.data.shape[0])
print("number of features per row (columns): ", dataset.data.shape[1])
#print(dataset.data[:4])

import pandas as pd
data_df = pd.DataFrame(dataset.data)

data_df.columns = ["MedInc", "HouseAge", "AveRooms",
                   "AveBedrms", "Population", "AveOccup",
                   "Latitude", "Longitude"]


print(data_df.head(5))

from sklearn.model_selection import train_test_split
data_sets = train_test_split(dataset.data,
                             dataset.target,
                             test_size=0.30,
                             random_state=42)

data_train, data_test, targets_train, targets_test = data_sets

clf = MLPRegressor(solver='lbfgs',        #  ‘lbfgs’, ‘sgd’, ‘adam’ (default)
                   alpha=1e-5,            # used for regularization, ovoiding overfitting by penalizing large magnitudes
                   hidden_layer_sizes=(10, 2),
                   activation='logistic', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’ (default)
                   max_iter=10000,
                   random_state=42)
clf.fit(data_train, targets_train)
print(clf.predict(data_train))
print(clf.predict(data_test))
print(len(data_train))
print(len(data_test))

print(clf.score(data_train, targets_train))
print(clf.score(data_test, targets_test))


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


exit()


import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
data, targets = make_regression(n_samples=100, n_features=1,noise=0.1)

print(data[:5])



from sklearn import datasets

plt.scatter(data, targets)
plt.show()

data, targets = make_regression(n_samples=100,
                                n_features=3,
                                #shuffle=True,
                                noise=0.1)

print(data[:5])

import pandas as pd
data_df = pd.DataFrame(data)
data_df.insert(len(data_df.columns),
               column="result",
               value=targets)
data_df.columns = "blue", "green", "red", "result"

print(data_df.head(5))


from sklearn.model_selection import train_test_split
data_sets = train_test_split(data,
                             targets,
                             test_size=0.30,
                             random_state=42)

data_train, data_test, targets_train, targets_test = data_sets


clf = MLPRegressor(solver='lbfgs',        #  ‘lbfgs’, ‘sgd’, ‘adam’ (default)
                   alpha=1e-5,            # used for regularization, ovoiding overfitting by penalizing large magnitudes
                   hidden_layer_sizes=(3, 1),
                   activation='logistic', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’ (default)
                   max_iter=10000,
                   random_state=42)
clf.fit(data_train, targets_train)
print(clf.predict(data_train[1:10]))
print(targets_train[1:10])