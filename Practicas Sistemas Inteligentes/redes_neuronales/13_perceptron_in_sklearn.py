import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

n_samples = 500
data, labels = make_blobs(n_samples=n_samples,
                             centers=([1.1, 3], [4.5, 6.9], [-1, 7]),
                             cluster_std=1.3,
                             random_state=0)


colours = ('green', 'orange', 'blue')
fig, ax = plt.subplots()

for n_class in range(3):
    ax.scatter(data[labels==n_class][:, 0],
               data[labels==n_class][:, 1],
               c=colours[n_class],
               s=50,
               label=str(n_class))

#plt.show()
iris = load_iris()
data = iris.data
labels = iris.target

datasets = train_test_split(data, labels, random_state=42, test_size=0.2)
train_data, test_data, train_labels, test_labels = datasets

p = Perceptron(random_state=42, max_iter=30, tol=0.001)
#p = KNeighborsClassifier()
p.fit(train_data, train_labels)

predictions_train = p.predict(train_data)
predictions_test = p.predict(test_data)
train_score = accuracy_score(predictions_train, train_labels)
print("score on train data: ", train_score)
test_score = accuracy_score(predictions_test, test_labels)
print("score on test data: ", test_score)

print(p.score(train_data, train_labels))
print(p.score(test_data, test_labels))

print(predictions_test)

print(classification_report(p.predict(train_data), train_labels))
print(classification_report(p.predict(test_data), test_labels))