from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
#y = [0, 0, 0, 1]
y = [0, 1, 1, 0]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 3), random_state=1)

print(clf.fit(X, y))

Lista = [0, 0], [0, 1], [1, 0], [0, 1], [1, 1], [2., 2.], [0.1, 0.9], [0.9, 0.1]
results = clf.predict(Lista)

for i in range(len(Lista)):
    print(Lista[i], results[i])

exit()

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

n_samples = 200
blob_centers = ([1, 1], [3, 4], [1, 3.3], [3.5, 1.8])
data, labels = make_blobs(n_samples=n_samples,
                          centers=blob_centers,
                          cluster_std=0.5,
                          random_state=0)


colours = ('green', 'orange', "blue", "magenta")
fig, ax = plt.subplots()

for n_class in range(len(blob_centers)):
    ax.scatter(data[labels==n_class][:, 0],
               data[labels==n_class][:, 1],
               c=colours[n_class],
               s=30,
               label=str(n_class))

from sklearn.model_selection import train_test_split
datasets = train_test_split(data,
                            labels,
                            test_size=0.2)

train_data, test_data, train_labels, test_labels = datasets

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(6,),
                    random_state=1)

clf.fit(train_data, train_labels)

clf.score(train_data, train_labels)

from sklearn.metrics import accuracy_score

predictions_train = clf.predict(train_data)
predictions_test = clf.predict(test_data)
train_score = accuracy_score(predictions_train, train_labels)
print("score on train data: ", train_score)
test_score = accuracy_score(predictions_test, test_labels)
print("score on test data: ", test_score)

print(predictions_train[:20])