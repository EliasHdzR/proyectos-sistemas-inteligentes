import numpy as np
from collections import Counter


class Perceptron:

    def __init__(self,
                 weights,
                 bias=1,
                 learning_rate=0.3):
        """
        'weights' can be a numpy array, list or a tuple with the
        actual values of the weights. The number of input values
        is indirectly defined by the length of 'weights'
        """
        self.weights = np.array(weights)
        self.bias = bias
        self.learning_rate = learning_rate

    @staticmethod
    def unit_step_function(x):
        if x <= 0:
            return 0
        else:
            return 1

    def __call__(self, in_data):
        """
                Perform forward pass through the perceptron.
        """
        in_data = np.concatenate((in_data, [self.bias]))
        result = self.weights @ in_data
        return Perceptron.unit_step_function(result)

    def adjust(self,
               target_result,
               in_data):
        """
        Adjust weights based on error.

        Args:
            target_result: Target output.
            in_data: Input data.
        """

        in_data = np.array(in_data)
        calculated_result = self(in_data)
        error = target_result - calculated_result
        if error != 0:
            in_data = np.concatenate((in_data, [self.bias]))
            correction = error * in_data * self.learning_rate
            self.weights += correction

    def evaluate(self, data, labels):
        """
        Evaluate the perceptron on a dataset.

        Args:
            data: List of input data arrays.
            labels: List of corresponding labels (0 or 1).

        Returns:
            Counter object containing evaluation results.
        """
        evaluation = Counter()
        for sample, label in zip(data, labels):
            result = self(sample)  # predict
            if result == label:
                evaluation["correct"] += 1
            else:
                evaluation["wrong"] += 1
        return evaluation