from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

centers = [[2, 3], [5, 5], [1, 8]]
n_classes = len(centers)
data, labels = make_blobs(n_samples=150,
                          centers=np.array(centers),
                          random_state=1)


from sklearn.datasets import load_digits
digits = load_digits()

colours = ('green', 'red', 'blue')
n_classes = 3
fig, ax = plt.subplots()
for n_class in range(0, n_classes):
    ax.scatter(data[labels==n_class, 0], data[labels==n_class, 1],
               c=colours[n_class], s=10, label=str(n_class))

ax.legend(loc='upper right')

#plt.show()

###################################################

digits = load_digits()
data = digits.data
labels = digits.target

from sklearn.model_selection import train_test_split
res = train_test_split(data, labels, train_size=0.8, random_state=1)

train_data, test_data, train_labels, test_labels = res

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)

# Instantiate the RadiusNeighborsClassifier
rnc = RadiusNeighborsClassifier(radius=50.0)
rnc.fit(train_data, train_labels)


predictedA = knn.predict(test_data)
#print("Predictions from the classifier:")
#print(predictedA)
#print("Target values:")
#print(test_labels)
print("Precision Prueba:", accuracy_score(predictedA, test_labels))
print("Precision Prueba:", accuracy_score(predictedA, test_labels, normalize=False))


predictedB = rnc.predict(test_data)
#print("Predictions from the classifier:")
#print(predictedB)
#print("Target values:")
#print(train_labels)
print("Precision Entrenamiento:", accuracy_score(predictedB, test_labels))
print("Precision Entrenamiento:", accuracy_score(predictedB, test_labels, normalize=False))


cmA = confusion_matrix(predictedA, test_labels)
print(cmA)
cmB = confusion_matrix(predictedB, test_labels)
print(cmB)
