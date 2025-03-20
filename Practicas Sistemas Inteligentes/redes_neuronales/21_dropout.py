import numpy as np
import random
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

import numpy as np
import matplotlib.pyplot as plt

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "../resources/"
train_data = np.loadtxt(data_path + "mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv",
                       delimiter=",")
print(test_data[:10])

print(test_data[test_data==255])
print(test_data.shape)

fac = 0.99 / 255
train_imgs = np.asarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asarray(train_data[:, :1])
test_labels = np.asarray(test_data[:, :1])

import numpy as np

lr = np.arange(10)

for label in range(10):
    one_hot = (lr==label).astype(np.int8)
    print("label: ", label, " in one-hot representation: ", one_hot)

lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float64)
test_labels_one_hot = (lr==test_labels).astype(np.float64)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 bias=None
                 ):

        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        X = truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)

        bias_node = 1 if self.bias else 0

        n = (self.no_of_in_nodes + bias_node) * self.no_of_hidden_nodes
        X = truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)
        self.wih = X.rvs(n).reshape((self.no_of_hidden_nodes,
                                     self.no_of_in_nodes + bias_node))

        n = (self.no_of_hidden_nodes + bias_node) * self.no_of_out_nodes
        X = truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)
        self.who = X.rvs(n).reshape((self.no_of_out_nodes,
                                     (self.no_of_hidden_nodes + bias_node)))

    def dropout_weight_matrices(self,
                                active_input_percentage=0.70,
                                active_hidden_percentage=0.70):
        # restore wih array, if it had been used for dropout
        self.wih_orig = self.wih.copy()
        self.no_of_in_nodes_orig = self.no_of_in_nodes
        self.no_of_hidden_nodes_orig = self.no_of_hidden_nodes
        self.who_orig = self.who.copy()

        active_input_nodes = int(self.no_of_in_nodes * active_input_percentage)
        active_input_indices = sorted(random.sample(range(0, self.no_of_in_nodes),
                                                    active_input_nodes))
        active_hidden_nodes = int(self.no_of_hidden_nodes * active_hidden_percentage)
        active_hidden_indices = sorted(random.sample(range(0, self.no_of_hidden_nodes),
                                                     active_hidden_nodes))

        self.wih = self.wih[:, active_input_indices][active_hidden_indices]
        self.who = self.who[:, active_hidden_indices]

        self.no_of_hidden_nodes = active_hidden_nodes
        self.no_of_in_nodes = active_input_nodes
        return active_input_indices, active_hidden_indices

    def weight_matrices_reset(self,
                              active_input_indices,
                              active_hidden_indices):

        """
        self.wih and self.who contain the newly adapted values from the active nodes.
        We have to reconstruct the original weight matrices by assigning the new values
        from the active nodes
        """

        temp = self.wih_orig.copy()[:, active_input_indices]
        temp[active_hidden_indices] = self.wih
        self.wih_orig[:, active_input_indices] = temp
        self.wih = self.wih_orig.copy()

        self.who_orig[:, active_hidden_indices] = self.who
        self.who = self.who_orig.copy()
        self.no_of_in_nodes = self.no_of_in_nodes_orig
        self.no_of_hidden_nodes = self.no_of_hidden_nodes_orig

    def train_single(self, input_vector, target_vector):
        """
        input_vector and target_vector can be tuple, list or ndarray
        """

        if self.bias:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.wih, input_vector)
        output_vector_hidden = activation_function(output_vector1)

        if self.bias:
            output_vector_hidden = np.concatenate((output_vector_hidden, [[self.bias]]))

        output_vector2 = np.dot(self.who, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.who += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1, :]
        else:
            x = np.dot(tmp, input_vector.T)
        self.wih += self.learning_rate * x

    def train(self, data_array,
              labels_one_hot_array,
              labels,
              epochs=1,
              active_input_percentage=0.70,
              active_hidden_percentage=0.70,
              no_of_dropout_tests=10):

        partition_length = int(len(data_array) / no_of_dropout_tests)

        for epoch in range(epochs):
            print("epoch: ", epoch)
            for start in range(0, len(data_array), partition_length):
                active_in_indices, active_hidden_indices = \
                    self.dropout_weight_matrices(active_input_percentage,
                                                 active_hidden_percentage)
                for i in range(start, start + partition_length):
                    self.train_single(data_array[i][active_in_indices],
                                      labels_one_hot_array[i])

                self.weight_matrices_reset(active_in_indices, active_hidden_indices)

            corrects, wrongs = self.evaluate(data_array, labels)
            print("accuracy train: ", corrects / (corrects + wrongs))

    def confusion_matrix(self, data_array, labels):
        cm = {}
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            if (target, res_max) in cm:
                cm[(target, res_max)] += 1
            else:
                cm[(target, res_max)] = 1
        return cm

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate((input_vector, [self.bias]))
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector, [[self.bias]]))

        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

parts = 10
partition_length = int(len(train_imgs) / parts)
print(partition_length)

start = 0
for start in range(0, len(train_imgs), partition_length):
    print(start, start + partition_length)

epochs = 3

L = [1.0, 0.75]
for d in L:
    simple_network = NeuralNetwork(no_of_in_nodes=image_pixels,
                               no_of_out_nodes=10,
                               no_of_hidden_nodes=100,
                               learning_rate=0.1)


    print("Using: ", d)
    simple_network.train(train_imgs,
                         train_labels_one_hot,
                         train_labels,
                         active_input_percentage=d,
                         active_hidden_percentage=d,
                         no_of_dropout_tests=100,
                         epochs=epochs)

corrects, wrongs = simple_network.evaluate(train_imgs, train_labels)
print("accuracy train: ", corrects / ( corrects + wrongs))
corrects, wrongs = simple_network.evaluate(test_imgs, test_labels)
print("accuracy: test", corrects / ( corrects + wrongs))