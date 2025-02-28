import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

centers = [[2, 3], [4, 5], [7, 9]]
data, labels = make_blobs(n_samples=1000,
                          centers=np.array(centers),
                          random_state=1)

labels[:7]

# some more blobs
fig, ax = plt.subplots()

colours = ('green', 'orange', 'blue')
for label in range(len(centers)):
    ax.scatter(x=data[labels==label, 0],
               y=data[labels==label, 1],
               c=colours[label],
               s=40,
               label=label)

ax.set(xlabel='X',
       ylabel='Y',
       title='Blobs Examples')


ax.legend(loc='upper right')

plt.show()


labels = labels.reshape((labels.shape[0],1))
all_data = np.concatenate((data, labels), axis=1)
np.savetxt("../resources/squirrels.txt", all_data, fmt=['%.3f', '%.3f', '%1d'])
exit()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

n_classes = 4
data, labels = make_blobs(n_samples=1000,
                          centers=n_classes,
                          random_state=100)

for i in range(10):
    print(data[i], labels[i])

# some blobs
fig, ax = plt.subplots()

colours = ('green', 'orange', 'blue', "pink")
for label in range(n_classes):
    ax.scatter(x=data[labels==label, 0],
               y=data[labels==label, 1],
               c=colours[label],
               s=40,
               label=label)

ax.set(xlabel='X',
       ylabel='Y',
       title='Blobs Examples')


ax.legend(loc='upper right')
plt.show()