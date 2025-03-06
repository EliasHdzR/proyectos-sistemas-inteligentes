import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
from sklearn.datasets import load_iris

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,
                                         self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can be tuples, lists or ndarrays
        """
        # make sure that the vectors have the right shape
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)
        target_vector = np.array(target_vector).reshape(target_vector.size, 1)

        output_vector_hidden = activation_function(self.weights_in_hidden @ input_vector)
        output_vector_network = activation_function(self.weights_hidden_out @ output_vector_hidden)

        output_error = target_vector - output_vector_network
        # calculate hidden errors:
        hidden_errors = self.weights_hidden_out.T @ output_error

        tmp = output_error * output_vector_network * (1.0 - output_vector_network)
        self.weights_hidden_out += self.learning_rate * (tmp @ output_vector_hidden.T)

        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        self.weights_in_hidden += self.learning_rate * (tmp @ input_vector.T)

    def run(self, input_vector):
        """
        running the network with an input vector 'input_vector'.
        'input_vector' can be tuple, list or ndarray
        """
        # make sure that input_vector is a column vector:
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)
        input4hidden = activation_function(self.weights_in_hidden @ input_vector)
        output_vector_network = activation_function(self.weights_hidden_out @ input4hidden)
        return output_vector_network

    def evaluate(self, data, labels):
        """
        Counts how often the actual result corresponds to the
        target result.
        A result is considered to be correct, if the index of
        the maximal value corresponds to the index with the "1"
        in the one-hot representation,
        e.g.
        res = [0.1, 0.132, 0.875]
        labels[i] = [0, 0, 1]
        """
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i].argmax():
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

from sklearn.datasets import make_blobs

n_samples = 500
blob_centers = ([2, 6], [6, 2], [7, 7])
n_classes = len(blob_centers)

iris = load_iris()
data = iris.data
labels = iris.target

import matplotlib.pyplot as plt

colours = ('green', 'red', "yellow")
fig, ax = plt.subplots()

for n_class in range(n_classes):
    ax.scatter(data[labels==n_class][:, 0],
               data[labels==n_class][:, 1],
               c=colours[n_class],
               s=40,
               label=str(n_class))


import numpy as np

labels = np.arange(n_classes) == labels.reshape(labels.size, 1)
labels = labels.astype(np.float64)

from sklearn.model_selection import train_test_split

res = train_test_split(data, labels, train_size=0.8, test_size=0.2, random_state=42)
train_data, test_data, train_labels, test_labels = res
#print(train_labels[:7])

simple_network = NeuralNetwork(no_of_in_nodes=4,
                               no_of_out_nodes=3,
                               no_of_hidden_nodes=7,
                               learning_rate=0.3)

for i in range(len(train_data)):
    simple_network.train(train_data[i], train_labels[i])

#print(simple_network.evaluate(train_data, train_labels))
#print(simple_network.evaluate(test_data, test_labels))

from neural_networks2 import NeuralNetwork

simple_network = NeuralNetwork(no_of_in_nodes=4,
                               no_of_out_nodes=3,
                               no_of_hidden_nodes=5,
                               learning_rate=0.1,
                               bias=None)

for i in range(len(train_data)):
    simple_network.train(train_data[i], train_labels[i])

#print(simple_network.evaluate(train_data, train_labels))
#print(simple_network.evaluate(test_data, test_labels))


c = np.loadtxt("../resources//strange_flowers.txt", delimiter=" ")

data = c[:, :-1]
labels = c[:, -1]
n_classes = int(np.max(labels)) # in our case 1, ... 4
print(data[:5])
print(labels[:10])

labels_one_hot = np.arange(1, n_classes+1) == labels.reshape(labels.size, 1)
labels_one_hot = labels_one_hot.astype(np.float64)
print(labels_one_hot[:3])

res = train_test_split(data, labels_one_hot,
                       train_size=0.8,
                       test_size=0.2,
                       random_state=42)
train_data, test_data, train_labels, test_labels = res
print(train_labels[:10])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data) #  fit and transform
test_data = scaler.transform(test_data) #  transform

from neural_networks2 import NeuralNetwork

simple_network = NeuralNetwork(no_of_in_nodes=4,
                               no_of_out_nodes=4,
                               no_of_hidden_nodes=20,
                               learning_rate=0.1,
                               bias=0.5)

for i in range(len(train_data)):
    simple_network.train(train_data[i], train_labels[i])

print(simple_network.evaluate(train_data, train_labels))
print(simple_network.evaluate(test_data, test_labels))