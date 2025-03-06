import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # necessary to reduce biases of large numbers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("../resources/strange_flowers.txt",
                      header=None,
                      names=["red", "green", "blue", "size", "label"],
                      sep=" ")
print(dataset)

import numpy as np

raw_data = np.loadtxt("../resources/strange_flowers.txt")
data = raw_data[:,:-1]
labels = raw_data[:,-1]

data = dataset.drop('label', axis=1)
labels = dataset.label

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

print(X_train[:5])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #  transform
X_test = scaler.transform(X_test) #  transform


print(X_train[:5])

k = int(len(X_train) ** 0.5)
print(k)

# Define the model
classifier = KNeighborsClassifier(n_neighbors=k,metric="minkowski",p=2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

# Evaluate Model
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(accuracy_score(y_test, y_pred))