import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "../resources/"
train_data = np.loadtxt(data_path + "mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv",
                       delimiter=",")

test_data[test_data==255]
test_data.shape

fac = 0.99 / 255
train_imgs = np.asarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asarray(train_data[:, :1])
test_labels = np.asarray(test_data[:, :1])

no_of_different_labels = 10
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
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


class NeuralNetwork:

    def __init__(self,
                 network_structure,  # ie. [input_nodes, hidden1_nodes, ... , hidden_n_nodes, output_nodes]
                 learning_rate,
                 bias=None
                 ):

        self.structure = network_structure
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        X = truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)

        bias_node = 1 if self.bias else 0
        self.weights_matrices = []
        layer_index = 1
        no_of_layers = len(self.structure)
        while layer_index < no_of_layers:
            nodes_in = self.structure[layer_index - 1]
            nodes_out = self.structure[layer_index]
            n = (nodes_in + bias_node) * nodes_out
            rad = 1 / np.sqrt(nodes_in)
            X = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)
            wm = X.rvs(n).reshape((nodes_out, nodes_in + bias_node))
            self.weights_matrices.append(wm)
            layer_index += 1

    def train_single(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray

        no_of_layers = len(self.structure)
        input_vector = np.array(input_vector, ndmin=2).T

        layer_index = 0
        # The output/input vectors of the various layers:
        res_vectors = [input_vector]
        while layer_index < no_of_layers - 1:
            in_vector = res_vectors[-1]
            if self.bias:
                # adding bias node to the end of the 'input'_vector
                in_vector = np.concatenate((in_vector,
                                            [[self.bias]]))
                res_vectors[-1] = in_vector
            x = np.dot(self.weights_matrices[layer_index], in_vector)
            out_vector = activation_function(x)
            res_vectors.append(out_vector)
            layer_index += 1

        layer_index = no_of_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T
        # The input vectors to the various layers
        output_errors = target_vector - out_vector
        while layer_index > 0:
            out_vector = res_vectors[layer_index]
            in_vector = res_vectors[layer_index - 1]

            if self.bias and not layer_index == (no_of_layers - 1):
                out_vector = out_vector[:-1, :].copy()

            tmp = output_errors * out_vector * (1.0 - out_vector)
            tmp = np.dot(tmp, in_vector.T)

            # if self.bias:
            #    tmp = tmp[:-1,:]

            self.weights_matrices[layer_index - 1] += self.learning_rate * tmp

            output_errors = np.dot(self.weights_matrices[layer_index - 1].T,
                                   output_errors)
            if self.bias:
                output_errors = output_errors[:-1, :]
            layer_index -= 1

    def train(self, data_array, labels_one_hot_array, labels, epochs=1, intermediate_results=False):
        intermediate_weights = []
        for epoch in range(epochs):
            for i in range(len(data_array)):
                self.train_single(data_array[i], labels_one_hot_array[i])
            if intermediate_results:
                intermediate_weights.append((self.wih.copy(), self.who.copy()))

            corrects, wrongs = self.evaluate(data_array, labels)
            print("accuracy train: ", corrects / (corrects + wrongs))

        return intermediate_weights


    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray

        no_of_layers = len(self.structure)
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))
        in_vector = np.array(input_vector, ndmin=2).T

        layer_index = 1
        # The input vectors to the various layers
        while layer_index < no_of_layers:
            x = np.dot(self.weights_matrices[layer_index - 1],
                       in_vector)
            out_vector = activation_function(x)

            # input vector for next layer
            in_vector = out_vector
            if self.bias:
                in_vector = np.concatenate((in_vector,
                                            [[self.bias]]))

            layer_index += 1

        return out_vector

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


epochs = 3

ANN = NeuralNetwork(network_structure=[image_pixels, 15, 15, 15,  10],
                    learning_rate=0.01,
                    bias=None)

ANN.train(train_imgs, train_labels_one_hot, train_labels, epochs=epochs)

corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
print("accuracy train: ", corrects / ( corrects + wrongs))
corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
print("accuracy: test", corrects / ( corrects + wrongs))

