from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

iris = load_iris()
data, labels = iris.data, iris.target

res = train_test_split(data, labels,
                       train_size=0.8,
                       test_size=0.2,
                       random_state=42)
train_data, test_data, train_labels, test_labels = res

print(train_data.shape)
print(test_data.shape)

n = 7
print(f"The first {n} data sets:")
print(test_data[:7])
print(f"The corresponding {n} labels:")
print(test_labels[:7])

fig, ax = plt.subplots()
# plotting learn data
colours = ('green', 'blue')
for n_class in range(2):
    ax.scatter(train_data[train_labels == n_class][:, 0],
               train_data[train_labels == n_class][:, 1],
               c=colours[n_class], s=40, label=str(n_class))

# plotting test data
colours = ('lightgreen', 'lightblue')
for n_class in range(2):
    ax.scatter(test_data[test_labels == n_class][:, 0],
               test_data[test_labels == n_class][:, 1],
               c=colours[n_class], s=40, label=str(n_class))

ax.plot()
plt.show()

exit()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
file_data = np.loadtxt("../resources/squirrels.txt")

data = file_data[:,:-1]
labels = file_data[:,-1]

data_sets = train_test_split(data,
                       labels,
                       train_size=0.25,
                       test_size=0.75,
                       random_state=42 # garantees same output for every run
                      )

train_data, test_data, train_labels, test_labels = data_sets

#print(data.shape, "\n",train_data.shape, "\n",test_data.shape)
print(iris.target)

indices = np.random.permutation(len(iris.data))
print(indices,"\n")

n_test_samples = 12
learnset_data = iris.data[indices[:-n_test_samples]]
learnset_labels = iris.target[indices[:-n_test_samples]]
testset_data = iris.data[indices[-n_test_samples:]]
testset_labels = iris.target[indices[-n_test_samples:]]
print(learnset_data[:4], learnset_labels[:4])
print(testset_data[:4], testset_labels[:4])