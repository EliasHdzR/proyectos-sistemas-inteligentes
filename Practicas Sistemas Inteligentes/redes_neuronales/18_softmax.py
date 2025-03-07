from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from neural_networks_softmax import NeuralNetwork

n_samples = 500
Lista = [[0, 0], [0, 5], [5, 0], [5, 5], [10, 5], [10, 10]]

samples, labels = make_blobs(n_samples=n_samples,
                             centers=(Lista),
                             random_state=0)

NumeroClases = len(Lista)
colours = ('green', 'red', 'blue', 'magenta', 'yellow', 'cyan')
fig, ax = plt.subplots()

for n_class in range(NumeroClases):
    ax.scatter(samples[labels == n_class][:, 0], samples[labels == n_class][:, 1],
               c=colours[n_class], s=40, label=str(n_class))

size_of_learn_sample = int(n_samples * 0.8)
size_of_test_sample = n_samples - size_of_learn_sample

learn_data = samples[:size_of_learn_sample]
test_data = samples[-size_of_test_sample:]

learn_labels = labels[:size_of_learn_sample]
test_labels = labels[-size_of_test_sample:]

plt.show()

Lx = [True, False]

for e in Lx:
    simple_network = NeuralNetwork(no_of_in_nodes=2,
                                   no_of_out_nodes=NumeroClases,
                                   no_of_hidden_nodes=5,
                                   learning_rate=0.3,
                                   softmax=e)

    #labels_one_hot = (np.arange(2) == labels.reshape(labels.size, 1))
    labels_one_hot = (np.arange(NumeroClases) == learn_labels.reshape(size_of_learn_sample, 1))
    labels_one_hot = labels_one_hot.astype(np.float64)
    print(labels_one_hot)

    for i in range(size_of_learn_sample):
        # print(learn_data[i], labels[i], labels_one_hot[i])
        simple_network.train(learn_data[i], labels_one_hot[i])

    from collections import Counter

    evaluation = Counter()
    #print(len(learn_data))
    #print(len(test_data))

    print(simple_network.evaluate(learn_data, learn_labels))
    print(simple_network.evaluate(test_data, test_labels))