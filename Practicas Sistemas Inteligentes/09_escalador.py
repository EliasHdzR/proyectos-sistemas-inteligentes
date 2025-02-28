import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # necessary to reduce biases of large numbers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("resources/strange_flowers.txt",
                      header=None,
                      names=["red", "green", "blue", "size", "label"],
                      sep=" ")

data = dataset.drop("label", axis=1)
labels = dataset["label"]

X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    random_state=0,
                                                    test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #  transform
X_test = scaler.transform(X_test) #  transform

print(X_train[:5])
print(X_test[:5])

k = int(len(X_train) ** 0.5)

# Define the model
classifier = KNeighborsClassifier(n_neighbors=k,
                                  metric="minkowski",
                                  p=2,    # Euclidian
                                 )

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))